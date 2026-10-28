#!/usr/bin/env python
"""
Preprocess IBM AML (Anti-Money-Laundering) transaction datasets into the
DyGLib-style format that CTDG-SSM's `get_link_prediction_data` expects.

Input   : LI-*-Trans.csv + LI-*-Patterns.txt   (raw IBM AML release)
Output  : processed_data/<name>/
    ml_<name>.csv        columns u, i, ts, label, idx   (idx runs 1..N, time-ordered)
    ml_<name>.npy        edge features [N+1, d] float32, ROW 0 = dummy zeros
                         (enron convention: array position == edge id)
    ml_<name>_node.npy   node features [num_nodes, d]  (log1p degrees), row k = node k
    edge_meta.csv        per-edge provenance: edge_id, laundering, in_patterns, schema_id,
                         attempt_id, leg_idx (position within its attempt), attempt_len
    meta.json            stats + config + vocabulary

Label semantics (verified on LI-Small):
    The CSV `Is Laundering` flag is a SUPERSET of the rows listed in Patterns.txt
    (all 1,023 pattern rows appear flagged in the CSV; the CSV additionally flags
    2,542 other scheme legs not printed in the Patterns file). We use the CSV
    flag as the binary `label` and keep the Patterns schema name as auxiliary
    metadata for the rows that appear there.

Memory: streams in chunks and remaps strings->ints per chunk; peak RSS ~2-3 GB.
Usage:
    python preprocess/preprocess_ibm_aml.py \
        --name li_small \
        --trans    /home/localadmin/aml_data/LI-Small_Trans.csv \
        --patterns /home/localadmin/aml_data/LI-Small_Patterns.txt
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

COLUMN_NAMES = [
    "Timestamp", "FromBank", "FromAccount", "ToBank", "ToAccount",
    "AmountReceived", "ReceivingCurrency", "AmountPaid",
    "PaymentCurrency", "PaymentFormat", "IsLaundering",
]


def parse_patterns(path: str):
    """Parse Patterns.txt into per-transaction metadata.

    Each `BEGIN/END LAUNDERING ATTEMPT - <schema>` block is ONE attempt; the
    listed transaction rows inside it are its legs, in file order (#1 = first
    leg, last = final leg). Returns
        key2schema : key -> schema name
        key2attempt: key -> attempt_id (1-based, BEGIN-block order)
        key2leg    : key -> leg index within its attempt (1-based)
        attempt_lengths : attempt_id -> total listed legs M
        schema_counts   : rows per schema
    """
    key2schema = {}
    key2attempt = {}
    key2leg = {}
    attempt_lengths = {}
    schema_counts = Counter()
    cur_schema = None
    attempt_id = 0
    cur_leg = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                attempt_id += 1
                cur_leg = 0
                cur_schema = line[len("BEGIN LAUNDERING ATTEMPT -"):].strip()
                continue
            if line.startswith("END LAUNDERING ATTEMPT"):
                attempt_lengths[attempt_id] = cur_leg
                cur_schema = None
                continue
            if cur_schema is None or line.count(",") != 10:
                continue
            key = ",".join(line.split(",")[:10])  # drop trailing flag col
            cur_leg += 1
            key2schema[key] = cur_schema
            key2attempt[key] = attempt_id
            key2leg[key] = cur_leg
            schema_counts[cur_schema] += 1
    return key2schema, key2attempt, key2leg, attempt_lengths, schema_counts


class IdMap:
    """Deterministic first-seen integer ids for string values, streaming."""

    def __init__(self):
        self.d = {}

    @property
    def next(self):
        return len(self.d)

    def codes(self, s: pd.Series) -> np.ndarray:
        """Map string series to int codes; first-seen values get 0,1,2,..."""
        mapped = s.map(self.d)          # C-level dict lookup
        isna = mapped.isna()
        if isna.any():
            unseen = s[isna].unique()
            for v in unseen:
                self.d[v] = len(self.d)
            mapped = mapped.copy()
            mapped[isna] = s[isna].map(self.d)
        return mapped.to_numpy(np.int64).astype(np.int32, copy=False)

    def names(self):
        inv = {v: k for k, v in self.d.items()}
        return [inv[i] for i in range(len(inv))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, required=True, help="dataset name e.g. li_small")
    ap.add_argument("--trans", type=str, required=True)
    ap.add_argument("--patterns", type=str, required=True)
    ap.add_argument("--out-root", type=str, default="processed_data")
    args = ap.parse_args()

    out = Path(args.out_root) / args.name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("[1/5] parsing patterns ...")
    key2schema, key2attempt, key2leg, attempt_lengths, schema_counts = parse_patterns(args.patterns)
    schema_names = list(schema_counts.keys())
    schema2id = {s: i for i, s in enumerate(schema_names)}
    n_attempts = len(attempt_lengths)
    print(f"      attempts {n_attempts} | pattern rows {sum(schema_counts.values())} | "
          f"unique keys {len(key2schema)} | schemas {len(schema_counts)} "
          f"| {time.time()-t0:.1f}s")

    print("[2/5] streaming transactions ...")
    AMAP, BMAP, FMAP = IdMap(), IdMap(), IdMap()   # accounts, banks, payment formats
    fmt_set, cur_set = set(), set()
    ts_c, src_c, dst_c, fm_c = [], [], [], []
    sban_c, scur_c, flag_c, pat_c, sch_c = [], [], [], [], []
    N = n_launder = n_pat_hit = n_pat_hit_flag = n_pat_hit_noflag = 0
    row_batches = []

    reader = pd.read_csv(
        args.trans, dtype=str, keep_default_na=False, chunksize=500_000,
        names=COLUMN_NAMES, header=0,
    )
    for ci, df in enumerate(reader):
        assert list(df.columns) == COLUMN_NAMES

        # ---- exact-match against Patterns (first 10 csv fields) ----
        key = df["Timestamp"].str.cat(
            df[["FromBank", "FromAccount", "ToBank", "ToAccount", "AmountReceived",
                "ReceivingCurrency", "AmountPaid", "PaymentCurrency", "PaymentFormat"]],
            sep=",",
        )
        schema = key.map(key2schema)              # str or NaN
        attempt = key.map(key2attempt).fillna(-1).to_numpy(np.int32)  # –1 = not in patterns
        leg = key.map(key2leg).fillna(-1).to_numpy(np.int32)
        in_pat = schema.notna().to_numpy()        # bool
        flag = df["IsLaundering"].eq("1").to_numpy()
        N += len(df)
        n_pat_hit += int(in_pat.sum())
        n_pat_hit_flag += int((in_pat & flag).sum())
        n_pat_hit_noflag += int((in_pat & ~flag).sum())
        n_launder += int(flag.sum())

        # ---- integer-coded columns (strings die at chunk end) ----
        ts = (pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
              .astype("int64").to_numpy() // 10**9)
        src = AMAP.codes(df["FromAccount"])
        dst = AMAP.codes(df["ToAccount"])
        fb = BMAP.codes(df["FromBank"])
        tb = BMAP.codes(df["ToBank"])
        fm = FMAP.codes(df["PaymentFormat"])
        sban = (fb == tb).astype(np.int8)
        scur = df["ReceivingCurrency"].eq(df["PaymentCurrency"]).to_numpy().astype(np.int8)
        amt_r = df["AmountReceived"].to_numpy(dtype=np.float64)
        amt_p = df["AmountPaid"].to_numpy(dtype=np.float64)
        schema_ids = schema.map(schema2id).fillna(-1).to_numpy(dtype=np.int16)

        fmt_set.update(df["PaymentFormat"].unique())
        cur_set.update(df["ReceivingCurrency"].unique())
        cur_set.update(df["PaymentCurrency"].unique())

        row_batches.append({
            "ts": ts, "src": src, "dst": dst, "fm": fm,
            "sban": sban, "scur": scur,
            "amt_r": amt_r, "amt_p": amt_p,
            "flag": flag.astype(np.int8), "in_pat": in_pat.astype(np.int8),
            "schema": schema_ids.astype(np.int16, copy=False),
            "attempt": attempt.astype(np.int32, copy=False),
            "leg": leg.astype(np.int32, copy=False),
        })
        if (ci + 1) % 4 == 0:
            tot = sum(len(b["ts"]) for b in row_batches)
            print(f"      chunk {ci+1} | rows {tot} | laundering {n_launder} | "
                  f"patterns-hits {n_pat_hit} | {time.time()-t0:.1f}s")
        del df, key, schema, in_pat, flag, ts, src, dst, fb, tb, fm, sban, scur, amt_r, amt_p, schema_ids

    ts = np.concatenate([b["ts"] for b in row_batches])
    src = np.concatenate([b["src"] for b in row_batches])
    dst = np.concatenate([b["dst"] for b in row_batches])
    fm = np.concatenate([b["fm"] for b in row_batches])
    sban = np.concatenate([b["sban"] for b in row_batches])
    scur = np.concatenate([b["scur"] for b in row_batches])
    amt_r = np.concatenate([b["amt_r"] for b in row_batches])
    amt_p = np.concatenate([b["amt_p"] for b in row_batches])
    flag = np.concatenate([b["flag"] for b in row_batches])
    in_pat = np.concatenate([b["in_pat"] for b in row_batches])
    schema = np.concatenate([b["schema"] for b in row_batches])
    attempt = np.concatenate([b["attempt"] for b in row_batches])
    leg = np.concatenate([b["leg"] for b in row_batches])
    del row_batches

    num_nodes = AMAP.next
    print(f"      rows {N} | nodes {num_nodes} | flagged {int(flag.sum())} | "
          f"in-patterns hits {n_pat_hit} (flagged {n_pat_hit_flag}, unflagged {n_pat_hit_noflag}) | "
          f"{time.time()-t0:.1f}s")

    print("[3/5] features + chronological sort ...")
    order = np.argsort(ts, kind="stable")
    ts, src, dst, fm, sban, scur, amt_r, amt_p, flag, in_pat, schema, attempt, leg = (
        ts[order], src[order], dst[order], fm[order], sban[order], scur[order],
        amt_r[order], amt_p[order], flag[order], in_pat[order], schema[order],
        attempt[order], leg[order],
    )
    # per-row attempt length (for "last-K legs" protocols)
    alen_arr = np.full(max(attempt_lengths.keys(), default=0) + 1, 0, dtype=np.int32)
    for aid, a_len in attempt_lengths.items():
        alen_arr[aid] = a_len
    attempt_len = np.where(attempt >= 0, alen_arr[np.maximum(attempt, 0)], 0)

    fmt_list = sorted(fmt_set)
    fmt_pos = {f: i for i, f in enumerate(fmt_list)}
    D_FMT = len(fmt_list)
    d_edge = 5 + D_FMT + 4

    feats = np.zeros((N, d_edge), dtype=np.float32)
    feats[:, 0] = np.log1p(amt_r)
    feats[:, 1] = np.log1p(amt_p)
    feats[:, 2] = np.log1p(np.abs(amt_r - amt_p))
    feats[:, 3] = scur.astype(np.float32)
    feats[:, 4] = sban.astype(np.float32)
    # sequential first-seen fmt codes -> sorted one-hot slots
    for f_val, code in FMAP.d.items():
        feats[fm == code, 5 + fmt_pos[f_val]] = 1.0
    hod = ts % 86400
    dow = (ts // 86400) % 7
    c0 = 5 + D_FMT
    feats[:, c0 + 0] = np.sin(2 * np.pi * hod / 86400.0)
    feats[:, c0 + 1] = np.cos(2 * np.pi * hod / 86400.0)
    feats[:, c0 + 2] = np.sin(2 * np.pi * dow / 7.0)
    feats[:, c0 + 3] = np.cos(2 * np.pi * dow / 7.0)
    assert np.isfinite(feats).all(), "non-finite feature value found"

    deg_out = np.bincount(src, minlength=num_nodes)
    deg_in = np.bincount(dst, minlength=num_nodes)
    node_feats = np.stack([np.log1p(deg_out), np.log1p(deg_in)], axis=1).astype(np.float32)

    print("[4/5] writing ml csv / npy ...")
    idx = np.arange(1, N + 1, dtype=np.int64)
    pd.DataFrame({
        "u": src, "i": dst, "ts": ts,
        "label": flag.astype(np.int32), "idx": idx,
    }).to_csv(out / f"ml_{args.name}.csv", index=False)

    edge_np = np.vstack([np.zeros((1, d_edge), dtype=np.float32), feats]).astype(np.float32)
    np.save(out / f"ml_{args.name}.npy", edge_np)
    np.save(out / f"ml_{args.name}_node.npy", node_feats)
    del edge_np, feats, amt_r, amt_p, fm, sban, scur

    print("[5/5] writing edge_meta.csv + meta.json ...")
    pd.DataFrame({
        "edge_id": idx,
        "laundering": flag.astype(np.int8),
        "in_patterns": in_pat,
        "schema_id": schema,
        "attempt_id": attempt,
        "leg_idx": leg,
        "attempt_len": attempt_len,
    }).to_csv(out / "edge_meta.csv", index=False)

    meta = {
        "name": args.name,
        "num_edges": int(N),
        "num_nodes": int(num_nodes),
        "ts_min": int(ts.min()),
        "ts_max": int(ts.max()),
        "ts_min_str": str(pd.Timestamp(int(ts.min()), unit="s")),
        "ts_max_str": str(pd.Timestamp(int(ts.max()), unit="s")),
        "n_laundering_flagged": int(flag.sum()),
        "n_pattern_rows": int(sum(schema_counts.values())),
        "n_attempts": n_attempts,
        "n_pattern_unique_keys": len(key2schema),
        "n_in_patterns_flagged": n_pat_hit_flag,
        "n_in_patterns_unflagged": n_pat_hit_noflag,
        "schema_row_counts": dict(schema_counts),
        "schema_names": schema_names,
        "payment_formats": fmt_list,
        "currencies": sorted(cur_set),
        "num_banks": BMAP.next,
        "feature_dims": d_edge,
        "feature_layout": (
            "0:log1p(amount_received) 1:log1p(amount_paid) 2:log1p(|rec-paid|) "
            "3:same_currency 4:same_bank -> one-hot payment_format -> sin/cos(hod)+sin/cos(dow)"
        ),
        "account_id_order": "first-seen (deterministic)",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"DONE in {time.time()-t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
