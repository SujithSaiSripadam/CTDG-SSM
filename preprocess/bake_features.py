#!/usr/bin/env python
"""
Bake the runtime edge features OFFLINE for IBM-AML datasets, so that
CTDG-SSM needs zero RAM-hungry work at training time.

Updates an existing processed_data/<name>/ folder produced by
`preprocess/preprocess_ibm_aml.py`:

  ml_<name>.npy        REWRITTEN: [N+1, 172] float32, pre-padded, and with the
                       exact features `add_edge_features` would produce at
                       runtime already baked in:
                         cols  0..d_raw-1  raw per-edge features (row 0 zeros)
                         cols  d_raw..+16  cos positional encoding of f2 (time enc)
                         col   last        f1 (pair cumulative count)
                       NOTE: baked with time_dim=16 (positional_encoding_mixer num_dims).
  ml_<name>_f2.npy     [N+1] float32, per-edge delta-t (f2), row 0 = 0
  ml_<name>_raw.npy    backup copy of the original [N+1, d_raw] features
  ml_<name>_node.npy   REWRITTEN: [num_nodes, 172] float32 (log degrees baked)

Rationale: at 6.9M edges, the in-loader padding (X->172 dims in *float64*) plus the
runtime add_edge_features hstack would peak at ~20 GB RAM on this 15 GB machine.
Pre-baking keeps everything float32 and skips both steps; the loader padding
branch short-circuits (shape[1] == 172 already).

Identical to runtime semantics because edge ids are 1..N in time-ordered CSV
rows and the ml npy rows are the same time order.
"""
import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

TIME_DIM = 16   # positional_encoding_mixer num_dims; must match args.time_dim at runtime


def positional_encoding_mixer(timestamps, num_dims):
    # exact copy of ssm_utlis.positional_encoding_mixer
    # so behavior is byte-identical to what the runtime would compute
    alpha = num_dims ** 0.5
    beta = num_dims ** 0.5
    i = np.arange(num_dims)[None]
    omega_raise = alpha ** ((-i + 1) / beta)
    pos = timestamps[:, None]
    return np.cos(pos * omega_raise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--out-root", type=str, default="processed_data")
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out_root) / args.name
    N1 = None

    # ---- load csv (just ts,u,i) -> time-sorted rows must match npy row order ----
    print("[1/5] reading ml csv (u,i,ts) ...")
    df = pd.read_csv(out / f"ml_{args.name}.csv", usecols=["u", "i", "ts"])
    N = len(df)
    N1 = N + 1
    src = df["u"].to_numpy(np.int64)
    dst = df["i"].to_numpy(np.int64)
    ts = df["ts"].to_numpy(np.int64)
    num_nodes = int(max(src.max(), dst.max())) + 1
    assert pd.read_csv(out / f"ml_{args.name}.csv", usecols=["idx"]).iloc[-1].item() == N

    print(f"      rows {N} | nodes {num_nodes}")

    # ---- raw npy: back up and load ----
    print("[2/5] back up raw features ...")
    raw_path = out / f"ml_{args.name}.npy"
    raw_backup = out / f"ml_{args.name}_raw.npy"
    if not raw_backup.exists():
        shutil.copy(raw_path, raw_backup)   # [N+1, d_raw] float32
    raw = np.load(raw_backup, mmap_mode="r")
    d_raw = raw.shape[1]
    assert raw.shape == (N1, d_raw), raw.shape

    # ---- compute f1 (per-pair cumulative count) and f2 (log1p gap) offline ----
    # exact replica of add_edge_features
    print("[3/5] computing f1/f2 (offline replica of add_edge_features) ...")
    # IMPORTANT: run_training normalizes full_data.node_interact_times /= 3600
    # BEFORE calling add_edge_features, so hour-scale ts must be used here.
    p = pd.DataFrame({
        "src": src, "dst": dst, "t": ts.astype(np.float64) / 3600.0,
    })
    p["pair"] = (p["src"] * num_nodes + p["dst"]).astype(np.int64)
    p["f1"] = p.groupby("pair", sort=False).cumcount()
    last = p.groupby("pair", sort=False)["t"].shift()
    f2 = p["t"].to_numpy() - last.to_numpy()
    f2 = np.where(np.isnan(f2), 1e11, f2)
    f2 = np.log1p(f2)                       # float64 -> [N]
    p_t = None
    print(f"      f1/f2 done | {time.time()-t0:.1f}s")

    # ---- time encoding ----
    print("[4/5] positional encoding of f2 ...")
    tenc = positional_encoding_mixer(f2, TIME_DIM)   # [N,16] float64
    f1 = p["f1"].to_numpy(np.int64)
    del p, last

    # ---- write baked 172-dim npy via memmap (no double allocation) ----
    print("[5/5] writing baked npy files ...")
    baked_cols = d_raw + TIME_DIM + 1       # 16 + 16 + 1 = 33
    mm = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float32,
                                   shape=(N1, 172))
    mm[:N1] = 0.0   # zero-fill once; col-slices below overwrite
    mm[1:N1, 0:d_raw] = np.asarray(raw[1:N1], dtype=np.float32)
    mm[1:N1, d_raw:d_raw + TIME_DIM] = tenc.astype(np.float32)
    mm[1:N1, -1] = f1.astype(np.float32)
    mm.flush()
    del mm

    f2mm = np.lib.format.open_memmap(out / f"ml_{args.name}_f2.npy", mode="w+",
                                     dtype=np.float32, shape=(N1,))
    f2mm[:N1] = 0.0
    f2mm[1:N1] = f2.astype(np.float32)
    f2mm.flush()
    del f2mm, tenc, f2, f1

    # ---- node npy baked to 172 dims ----
    deg_out = np.bincount(src, minlength=num_nodes)
    deg_in = np.bincount(dst, minlength=num_nodes)
    nmm = np.lib.format.open_memmap(out / f"ml_{args.name}_node.npy", mode="w+",
                                    dtype=np.float32, shape=(num_nodes, 172))
    nmm[:num_nodes] = 0.0
    nmm[:num_nodes, 0] = np.log1p(deg_out).astype(np.float32)
    nmm[:num_nodes, 1] = np.log1p(deg_in).astype(np.float32)
    nmm.flush()
    del nmm

    print(f"DONE baked features for '{args.name}' in {time.time()-t0:.1f}s")
    print(f"  ml_{args.name}.npy      -> [N+1=({N}+1), 172] float32  (pre-baked: raw|time-enc|f1)")
    print(f"  ml_{args.name}_f2.npy   -> [N+1] float32  (runtime LastInteraction)")
    print(f"  ml_{args.name}_node.npy -> [num_nodes, 172] float32")
    print(f"  ml_{args.name}_raw.npy  -> original raw npy backup")


if __name__ == "__main__":
    main()
