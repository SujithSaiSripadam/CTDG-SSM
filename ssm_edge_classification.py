"""
ssm_edge_classification.py / Edge (laundering) classification for CTDG-SSM.

Streams the transaction graph chronologically, and for every edge predicts whether
it is part of a laundering attempt (`Is Laundering` flag), using the SAME
memory/ssm machinery as link prediction: window slicing, neighbor sampling, the
per-window Laplacian, zoh_update after prediction, t_warm warm-up, and 10-window
gradient accumulation (all mirrored from ssm_learn.link_prediction_train_*).

=========================== DESIGN DECISIONS ===========================

1) LABELS / same as IBM/Multi-GNN (Egressy et al., AAAI'24 / NeurIPS'23):
   they simply use the `Is Laundering` column of the Trans csv as the binary
   edge label of every transaction. We already verified on LI-Small that this
   flag (3,565 edges) is a SUPERSET of the rows listed in Patterns.txt (1,023).
   We therefore follow their protocol exactly and use the CSV flag as `label`.
   `--label_source {flag, patterns}` restricts positives to the rows actually
   listed in Patterns.txt (via edge_meta.csv, keyed by edge_id) for ablations.

2) TEMPORAL BOUND / none beyond causality (same spirit as Multi-GNN):
   They embed each edge into a static graph and sample neighborhoods WITHOUT a
   time filter; splits are purely chronological. Our protocol is strictly more
   conservative: sampling uses ONLY neighbors with time <= window start (the
   tau filter in temporal_sample_neighbors), and memory is updated only after
   the window's predictions are made. This guarantees zero future leakage,
   while remaining "pattern agnostic": we do NOT bound classification windows
   to any laundering-attempt boundary.

3) SPLIT / Multi-GNN's day-boundary split:
   They choose two day-cut points that best match the target split proportions
   (their default 0.6/0.2/0.2), so no DAY straddles a split boundary. A scheme
   running across a boundary is split (accepted in their work too); since
   labels are attached per edge and memory only ever sees the stream causally,
   a straddling scheme is still a valid, solvable classification task in both
   halves. We use ALL edges of the dataset (unlike the LP loader, which drops
   edges for the 10% inductive-node holdout, Multi-GNN classifies on all
   transactions, so we read the full stream directly).

4) EVAL PROTOCOL / streaming, like every other loop in this repo:
   Per epoch: warm-up (t_warm windows, memory only) -> stream train windows
   (predict + BCE + zoh, 10-window gradient chunks) -> stream val windows
   (predict, no_grad, memory CONTINUES from train, matching Multi-GNN's val
   graph = train graph + val edges). Early stopping on validation F1
   (Multi-GNN's model-selection metric), with AP/AUC printed for reference
   because F1@0.5 is unstable at ~0.05% positive rate.
   Final test: reload best weights, reset, BURN-IN over the full train+val
   stream with zoh only, then stream test windows and predict. Test context
   graph (neighbor CSR) is built on train+val+test, matching Multi-GNN, but
   the tau time-filter keeps sampling causal.
   NOTE (LP convention kept): eval loops stop 2*T windows before the end of
   the stream, exactly like link_prediction_eval in ssm_learn.py.

5) IMBALANCE / BCEWithLogits with configurable positive-class weight:
   `--pos_weight auto` uses n_neg/n_pos, `sqrt` uses sqrt(n_neg/n_pos), and a
   numeric value (for example `1.0`) sets a fixed weight.

6) WARM-UP / t_warm=100 windows (same as LP):
   The first t_warm windows of every train epoch are used ONLY to fill memory
   (zoh), no predictions/gradients; mirrors ssm_learn LP exactly. During the
   final test, burn-in plays the same role over train+val.

7) First/last legs under the DEFAULT (flag) protocol: we do NOT special-case
   the first or last edge of a pattern; every listed leg is just a positive
   edge. Decision 8 adds the temporal late-legs alternative; the
   attempt_id/leg_idx metadata is now IN edge_meta.csv, so a per-leg-position
   diagnostic (first/mid/last F1) is straightforward future analysis.

8) LATE-LEGS PROTOCOL (--label_source late, --late_legs K, default 5) —
   the temporal answer to "when can an SSM actually see through a scheme?":
   ONLY the last K listed legs of each laundering attempt keep label +1;
   the EARLY legs of each attempt get label -1 = EXCLUDED: they are still
   streamed through memory (the transactions happened — causal context must
   not be skipped) but contribute nothing to the loss or the metrics.
   Every non-listed edge stays a background negative (label 0).
   Requires edge_meta.csv columns attempt_id/leg_idx/attempt_len (written by
   preprocess_ibm_aml.py >= this version). For Multi-GNN-comparable numbers
   use --label_source flag (all legs positive).

========================================================================

Usage (mirrors ssm_link_prediction.py):
    python ssm_edge_classification.py --dataset li_small --device cuda \
        --epochs 2 --runs 1 --batches 128 --pos_weight auto --name ec_exp1
    # quick smoke of the first 40 windows of every pass:
    python ssm_edge_classification.py --dataset li_small --smoke 40
"""

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("Changed working directory to:", os.getcwd())

import gc
import itertools
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ssm_dataset import TemporalEdgeDataset
from ssm_memory import MemoryModel
from ssm_utlis import positional_encoding_mixer, add_edge_features

from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, precision_recall_curve

import torch.optim as optim

IBM_AML_DATASETS = {'li_small', 'li_medium', 'li_large'}


def load_env():
    """Load KEY=VALUE pairs from CTDG-SSM/.env into os.environ (WANDB_API_KEY etc.).
    Uses python-dotenv when available, falls back to a tiny parser."""
    env_path = os.path.join(script_dir, '.env')
    if not os.path.exists(env_path):
        print(f"[./.env] not found at {env_path} — add WANDB_API_KEY=... there for wandb")
        return False
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        return True
    except Exception:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        return True


# ----------------------------------------------------------------------------
# Multi-GNN-style split: chronological DAY-boundary cuts matching target ratios
# (replica of their data_loading split search on daily counts; ts in seconds)
# ----------------------------------------------------------------------------
def multiggn_day_split(ts: np.ndarray, val_ratio: float, test_ratio: float):
    """Choose day-cut points i, j such that the three day-telescoping windows
    [0,i), [i,j), [j,end) best match (1-v-t, v, t) proportions; return edge masks."""
    rel = ts - ts.min()
    n_days = int(rel.max() / (3600 * 24)) + 1
    day_inds = []
    for day in range(n_days):
        day_inds.append(np.where((rel >= day * 86400) & (rel < (day + 1) * 86400))[0])

    target = (1.0 - val_ratio - test_ratio, val_ratio, test_ratio)
    daily_counts = np.array([len(x) for x in day_inds], dtype=np.float64)
    split_scores = {}
    for i, j in itertools.combinations(range(n_days), 2):
        split_totals = [daily_counts[:i].sum(), daily_counts[i:j].sum(), daily_counts[j:].sum()]
        split_props = np.array(split_totals) / max(np.sum(split_totals), 1)
        err = [abs(v - t) / t for v, t in zip(split_props, target)]
        split_scores[(i, j)] = max(err)
    cut_i, cut_j = min(split_scores, key=split_scores.get)

    train_mask = np.zeros(len(ts), dtype=bool)
    val_mask = np.zeros(len(ts), dtype=bool)
    test_mask = np.zeros(len(ts), dtype=bool)
    for day in range(0, cut_i):
        train_mask[day_inds[day]] = True
    for day in range(cut_i, cut_j):
        val_mask[day_inds[day]] = True
    for day in range(cut_j, n_days):
        test_mask[day_inds[day]] = True

    assert train_mask.sum() + val_mask.sum() + test_mask.sum() == len(ts)
    return train_mask, val_mask, test_mask, (cut_i, cut_j)


def load_all_edges(dataset_name, label_source='flag', late_legs=5):
    """Load the FULL chronological stream of every edge with its label.

    IBM-AML datasets (li_*): read processed_data/<name>/ml_<name>.csv directly.
    We deliberately do NOT go through get_link_prediction_data with 0/0 ratios:
    (a) with val_ratio=0 the loader's inductive-node machinery breaks, and
    (b) even at working ratios it silently drops edges touching the sampled
    10% new-node holdout — Multi-GNN classifies on ALL transactions, so we
    keep the complete stream (N rows: u, i, ts, idx 1..N, label).

    Other datasets: use the standard loader's full_data (already all rows).
    Labels:
      'flag'     -> CSV `Is Laundering` superset (like Multi-GNN): 0/1
      'patterns' -> edge_meta.csv: laundering==1 AND in_patterns==1 only: 0/1
      'late'     -> last `late_legs` legs of each attempt: +1; remaining
                    attempt legs: -1 (EXCLUDED from loss & metrics, still
                    streamed for memory); non-listed edges: 0
                    (needs attempt_id/leg_idx/attempt_len in edge_meta.csv)
    """
    if dataset_name in IBM_AML_DATASETS:
        df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
        full_src = df.u.values.astype(np.longlong)
        full_dst = df.i.values.astype(np.longlong)
        full_ts = df.ts.values.astype(np.float64)
        edge_ids = df.idx.values.astype(np.longlong)
        flag_labels = df.label.values.astype(np.int64) if 'label' in df.columns \
            else np.zeros(len(df), dtype=np.int64)

        meta = pd.read_csv('./processed_data/{}/edge_meta.csv'.format(dataset_name))
        meta = meta.set_index('edge_id')
        meta_idx = edge_ids - 1           # edge_ids are 1..N -> meta row index
        ok = meta.laundering.to_numpy().astype(bool)[meta_idx]
        in_pat = meta.in_patterns.to_numpy().astype(bool)[meta_idx]
        #pat_labels = (ok & in_pat).astype(np.int64) #for n-5 labels only
        # Positive = every transaction matched to Patterns.txt,
        # regardless of the CSV IsLaundering flag.
        pat_labels = in_pat.astype(np.int64)

        late_labels = None
        if {'attempt_id', 'leg_idx', 'attempt_len'}.issubset(meta.columns):
            aid = meta.attempt_id.to_numpy()[meta_idx]
            leg = meta.leg_idx.to_numpy()[meta_idx]
            a_len = meta.attempt_len.to_numpy()[meta_idx]
            hit = (aid > 0)
            # legs within the last K positions of their attempt
            is_late = hit & (leg > a_len - late_legs)
            late_labels = np.full(len(df), 0, dtype=np.int64)
            late_labels[is_late] = 1
            late_labels[hit & ~is_late] = -1    # early attempt legs: excluded

        node_raw_features = np.load(
            './processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name), mmap_mode='r')
        edge_raw_features = np.load(
            './processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name), mmap_mode='r')
        assert edge_raw_features.shape[0] == len(df) + 1
        num_nodes = node_raw_features.shape[0]
    else:
        from utils.DataLoader import get_link_prediction_data
        node_raw_features, edge_raw_features, full, *_ = get_link_prediction_data(
            dataset_name=dataset_name, val_ratio=0.15, test_ratio=0.15)
        # full_data keeps every row of the csv in chronological order
        full_src, full_dst = full.src_node_ids, full.dst_node_ids
        full_ts = full.node_interact_times.astype(np.float64)
        edge_ids = full.edge_ids
        flag_labels = np.asarray(full.labels).astype(np.int64)
        pat_labels = flag_labels           # no pattern metadata for non-IBM sets
        late_labels = flag_labels
        num_nodes = node_raw_features.shape[0]

    if label_source == 'flag':
        labels = flag_labels
    elif label_source == 'patterns':
        labels = pat_labels
    elif label_source == 'late':
        if late_labels is None:
            raise ValueError("label_source='late' needs attempt_id/leg_idx/attempt_len "
                             "in edge_meta.csv (re-run preprocess_ibm_aml.py)")
        labels = late_labels
    else:
        raise ValueError(f"unknown label_source {label_source}")
    n_excl = int((labels == -1).sum())
    print(f"[labels] source={label_source} -> edges={len(labels)} "
          f"positives={int((labels == 1).sum())} excluded(early legs)={n_excl}")

    from utils.DataLoader import Data
    full_data = Data(src_node_ids=full_src, dst_node_ids=full_dst,
                     node_interact_times=full_ts, edge_ids=edge_ids, labels=labels)
    return node_raw_features, edge_raw_features, full_data, num_nodes


def prepare_features(args, node_raw_features, edge_raw_features, full_data):
    """Baked route for IBM-AML datasets; otherwise compute add_edge_features at
    runtime exactly as ssm_link_prediction.py does (same dataset special cases)."""
    if args.dataset in IBM_AML_DATASETS:
        # Features are PRE-BAKED offline by preprocess/bake_features.py into
        # ml_<name>.npy as [N+1, 172] float32
        # (cols = raw | cos-time-enc(f2) | f1, row 0 = dummy zeros).
        assert edge_raw_features.shape[1] == 172
        tmx = np.asarray(np.load(   # per-edge delta-t (f2), [N+1] float32, edge_ids 1..N
            './processed_data/{}/ml_{}_f2.npy'.format(args.dataset, args.dataset),
            mmap_mode='r'))
        if getattr(args, 'prune_features', True):
            # Active features: cols 0..31 (raw + time_enc) and col 171 (f1 cumulative count).
            # Cols 32..170 (139 columns) are dummy zero padding from offline bake.
            print("[prepare_features] Pruning 139 zero columns (172 -> 33 active dimensions: raw=16, tenc=16, f1=1)")
            edge_raw_features = np.ascontiguousarray(
                np.hstack([edge_raw_features[:, :32], edge_raw_features[:, -1:]])
            )
    else:
        # replicate ssm_link_prediction.py runtime feature construction
        if args.dataset in ['enron', 'uci', 'lastfm']:
            edge_raw_features = np.asarray(edge_raw_features)
            edge_raw_features = edge_raw_features.mean(axis=-1)[:, None]
        elif args.dataset in ['mooc']:
            edge_raw_features = edge_raw_features[:, :4]
        elif args.dataset in ['SocialEvo']:
            edge_raw_features = edge_raw_features[:, :2]

        edge_raw_features, tmx_ = add_edge_features(
            full_data.src_node_ids,
            full_data.dst_node_ids,
            full_data.node_interact_times,
            np.asarray(edge_raw_features)[1:],
            positional_encoding_mixer,
            args.time_dim
        )
        zero_padd = np.zeros((1, edge_raw_features.shape[1]))
        edge_raw_features = np.vstack((zero_padd, edge_raw_features))
        fin_tmx = np.zeros((len(tmx_) + 1,))
        fin_tmx[1:] = tmx_
        tmx = fin_tmx
    return edge_raw_features, tmx


def compute_classification_metrics(labels, logits, eval_threshold=None):
    """Compute comprehensive edge classification metrics.

    Args:
        labels: 1D np.ndarray of int8 {0, 1}
        logits: 1D np.ndarray of float32
        eval_threshold: float or None. If float, evaluates at this fixed threshold
            (e.g. best threshold determined on validation set). If None, sweeps
            thresholds to find the optimal F1 and threshold.
            IMPORTANT: eval_threshold is expected in PROBABILITY space [0,1].

    Returns:
        metrics dict with:
          'F1@0.5': F1 at raw threshold 0.5 (logits > 0 <=> prob > 0.5)
          'F1_opt': optimal F1 (swept in probability space if eval_threshold is None,
                    else evaluated at eval_threshold)
          'F1': backward-compatible alias to F1_opt (best achievable F1)
          'best_thr': optimal threshold in LOGIT space (for direct application to logits)
          'best_thr_prob': optimal threshold in PROBABILITY space [0,1]
          'AP': Average Precision (PR-AUC)
          'AUC': ROC-AUC
    """
    if len(labels) == 0 or len(np.unique(labels)) <= 1:
        return {
            'F1@0.5': 0.0,
            'F1_opt': 0.0,
            'F1': 0.0,
            'best_thr': 0.0,
            'best_thr_prob': 0.5,
            'AP': 0.0,
            'AUC': 0.5,
            'degenerate': True
        }

    # Convert logits to probabilities for threshold sweeping
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    # 1. Raw threshold @0.5 (logits > 0 <=> prob > 0.5)
    preds_05 = (logits > 0).astype(np.int8)
    f1_05 = float(f1_score(labels, preds_05, zero_division=0))

    # 3. Optimal threshold — sweep in PROBABILITY space [0,1]
    if eval_threshold is None:
        p, r, th_prob = precision_recall_curve(labels, probs)
        # th_prob has length len(p)-1; thresholds are in probability space
        denom = p[:-1] + r[:-1]
        f1_arr = np.divide(2 * p[:-1] * r[:-1], denom, out=np.zeros_like(denom), where=denom > 0)
        if len(f1_arr) > 0:
            best_idx = int(np.argmax(f1_arr))
            best_f1 = float(f1_arr[best_idx])
            best_thr_prob = float(th_prob[best_idx])
        else:
            best_f1 = f1_05
            best_thr_prob = 0.5
    else:
        best_thr_prob = float(eval_threshold)
        preds_opt = (probs >= best_thr_prob).astype(np.int8)
        best_f1 = float(f1_score(labels, preds_opt, zero_division=0))

    # Convert best probability threshold back to logit space for direct use
    best_thr_prob = np.clip(best_thr_prob, 1e-7, 1 - 1e-7)
    best_thr = float(np.log(best_thr_prob / (1.0 - best_thr_prob)))

    # 4. Threshold-independent ranking metrics (use logits or probs, same result)
    ap = float(average_precision_score(labels, logits))
    auc = float(roc_auc_score(labels, logits))

    return {
        'F1@0.5': f1_05,
        'F1_opt': best_f1,
        'F1': best_f1,  # backward-compat: best achievable F1
        'best_thr': best_thr,
        'best_thr_prob': best_thr_prob,
        'AP': ap,
        'AUC': auc,
    }


class BinaryFocalLoss(nn.Module):
    """Binary Focal Loss with logits and optional pos_weight for extreme class imbalance.
    Downweights easy negative examples (gamma >= 1), while preserving full gradient
    on rare positives (asymmetric focal formulation to prevent positive gradient starvation)."""
    def __init__(self, gamma=2.0, pos_weight=1.0, reduction='mean'):
        super().__init__()
        self.gamma = float(gamma)
        self.pos_weight = float(pos_weight)
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        # Asymmetric modulation: modulate only negatives with probs^gamma,
        # leaving rare positives unmodulated (weight 1.0) so their gradient is never choked.
        focal_weight = targets * 1.0 + (1.0 - targets) * (probs ** self.gamma)
        if self.pos_weight != 1.0:
            alpha = targets * self.pos_weight + (1.0 - targets)
            loss = alpha * focal_weight * bce
        else:
            loss = focal_weight * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class EdgePredictionHead(nn.Module):
    """Prediction head for dynamic edge classification.
    Supports:
      - 'linear': Direct linear projection with LayerNorm.
      - 'mlp': 2-layer MLP with LayerNorm, GELU, and Dropout.
      - 'cross' / 'dcn': Explicit 2nd-order feature crossing (Deep & Cross Network style)
                         plus residual projection. Solves the 'Additive Trap' by allowing
                         transaction attributes (amount, currency, format) to explicitly
                         interact with source/destination node memory states (account history).
    """
    def __init__(self, in_features, hidden_dim, head_type='cross', dropout=0.1):
        super().__init__()
        self.head_type = head_type.lower()
        self.norm = nn.LayerNorm(in_features)

        if self.head_type == 'linear':
            self.net = nn.Linear(in_features, 1)
        elif self.head_type == 'mlp':
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, 1),
            )
        elif self.head_type in ['cross', 'dcn']:
            # Explicit degree-2 interaction layer:
            # x_cross = x_norm * (x_norm * cross_w + cross_b) + x_norm
            self.cross_w = nn.Parameter(torch.randn(in_features) * (1.0 / np.sqrt(in_features)))
            self.cross_b = nn.Parameter(torch.zeros(in_features))
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.skip = nn.Linear(in_features, 1)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, x):
        x_norm = self.norm(x)
        if self.head_type == 'linear':
            return self.net(x_norm)
        elif self.head_type == 'mlp':
            return self.net(x_norm)
        elif self.head_type in ['cross', 'dcn']:
            x_cross = x_norm * (x_norm * self.cross_w + self.cross_b) + x_norm
            return self.net(x_cross) + self.skip(x_norm)


def edge_model_with_head(args, num_nodes, edge_feat_dim, pw=1.0, n_pos_tr=1, n_neg_tr=1):
    model = MemoryModel(
        num_nodes=num_nodes, input_dim=edge_feat_dim, hidden_dim=args.hidden_dim,
        time_dim=args.time_dim, reg=args.reg, delta=args.delta, device=args.device,
        embd_dims=args.embd_dim, update_type='mamba', use_old_message=False
    ).to(args.device)
    use_relational = bool(getattr(args, 'use_relational', True))
    node_state_dim = model.embd_dims + model.hidden_dim
    if use_relational:
        # Enhancement 2A: Bilinear relational interactions (hadamard product + abs difference)
        in_features = 4 * node_state_dim + model.time_dim + 1 + edge_feat_dim
    else:
        in_features = 2 * node_state_dim + model.time_dim + 1 + edge_feat_dim

    head_type = getattr(args, 'head_type', 'cross').lower()
    edge_head = EdgePredictionHead(
        in_features=in_features,
        hidden_dim=args.hidden_dim,
        head_type=head_type,
        dropout=getattr(args, 'dropout', 0.1)
    ).to(args.device)

    # Initialize final layer bias to match class prior (log(n_pos/n_neg)).
    # We preserve PyTorch's default kaiming_uniform_ on weights to maintain feature diversity.
    prior_logit = -np.log(max(pw, 1.0))
    prior_logit = float(np.clip(prior_logit, -10.0, 0.0))

    with torch.no_grad():
        if edge_head.head_type == 'linear':
            if edge_head.net.bias is not None:
                nn.init.constant_(edge_head.net.bias, prior_logit)
        elif edge_head.head_type == 'mlp':
            if edge_head.net[-1].bias is not None:
                nn.init.constant_(edge_head.net[-1].bias, prior_logit)
        elif edge_head.head_type in ['cross', 'dcn']:
            if edge_head.net[-1].bias is not None:
                nn.init.constant_(edge_head.net[-1].bias, prior_logit * 0.5)
            if edge_head.skip.bias is not None:
                nn.init.constant_(edge_head.skip.bias, prior_logit * 0.5)

    edge_head.use_relational = use_relational
    return model, edge_head


def predict_edges(model, head, data):
    """Per-edge score. Concatenates src/dst node memory states, time encoding,
    and the current transaction's edge features (from data.x_sub[:E]).
    When head.use_relational=True, augments with bilinear element-wise product
    and absolute difference of node states.
    Read BEFORE the window's zoh_update so predictions are strictly causal."""
    E = data.target_src.numel()
    states_src = torch.hstack([model.names(data.target_src), model.ne_vec[data.target_src]])
    states_dst = torch.hstack([model.names(data.target_dst), model.ne_vec[data.target_dst]])
    time_enc = model.time_encoder(data.target_enc)
    edge_feats = data.x_sub[:E]
    if getattr(head, 'use_relational', False):
        prod = states_src * states_dst
        diff = torch.abs(states_src - states_dst)
        x = torch.hstack([states_src, states_dst, prod, diff, time_enc, edge_feats])
    else:
        x = torch.hstack([states_src, states_dst, time_enc, edge_feats])
    return head(x).squeeze(-1)


def edge_epoch_stream(model, head, dataset, optimizer=None, loss_fn=None,
                      eval_threshold=None, bptt_windows=25, neg_sample_rate=1.0,
                      log_odds_correction=False,
                      t_warm=100, smoke=None, desc=''):
    """One chronological pass over a TemporalEdgeDataset with task='NC'.

    train pass (optimizer is not None):
        t_warm memory-only windows, then predict+loss per window with memory
        updated via zoh_update inside the autograd graph (accum-through);
        gradient clipping and backward/step/model.detach() every bptt_windows
        (default 25; captures ~6,400 edges / ~25 minutes of continuous graph dynamics)
        or on the final `end_train` window.
        When neg_sample_rate < 1.0, negatives are subsampled for LOSS calculation ONLY
        while 100% of transactions still update Mamba state via zoh_update.
    eval pass:
        no_grad, memory continues. Evaluates dataset stream UP TO 2*T windows
        before the end (matching link_prediction_eval and node_classification_eval
        in ssm_learn.py / ssm_nc_learn.py) to avoid incomplete context windows.
    """
    train_mode = optimizer is not None
    if train_mode:
        model.train()
        head.train()
        optimizer.zero_grad()
    else:
        model.eval()
        head.eval()

    # Full stream for both train and eval (all edges are evaluated)
    if getattr(dataset, 'unique_batch', False):
        counter = range(len(dataset))
        if smoke is not None:
            counter = range(min(len(dataset), smoke))
    else:
        counter = range(0, len(dataset), dataset.T)
        if smoke is not None:
            counter = range(0, min(len(dataset), smoke * dataset.T), dataset.T)

    logits_all, labels_all = [], []   # GPU tensors; ONE .cpu() transfer at the end
    total_loss = 0.0
    n_loss_windows = 0
    valid_grad = False
    batch_loss = None
    batch_counter = 0

    for window_idx, t in enumerate(tqdm(counter, desc=desc, disable=False)):
        data = dataset[t]
        if train_mode and window_idx >= t_warm:
            logits = predict_edges(model, head, data)
            labels = data.label_t
            valid = labels != -1        # -1 = excluded edge (late-legs protocol)
            if valid.any():
                logits_v = logits[valid]
                labels_v = labels[valid]

                # Decoupled Loss-Memory Propagation: Subsample negatives for training loss
                if neg_sample_rate < 1.0:
                    pos_mask = (labels_v == 1)
                    neg_mask = (labels_v == 0)
                    if neg_mask.any():
                        neg_rand = torch.rand(neg_mask.sum(), device=labels_v.device)
                        neg_keep = neg_rand < neg_sample_rate
                        if not neg_keep.any():
                            neg_keep[torch.randint(0, len(neg_keep), (1,), device=labels_v.device)] = True
                        neg_idx = neg_mask.nonzero(as_tuple=True)[0][neg_keep]
                        pos_idx = pos_mask.nonzero(as_tuple=True)[0]
                        sel_idx = torch.cat([pos_idx, neg_idx])
                    else:
                        sel_idx = pos_mask.nonzero(as_tuple=True)[0]
                else:
                    sel_idx = torch.arange(len(labels_v), device=labels_v.device)

                if len(sel_idx) > 0:
                    logits_sel = logits_v[sel_idx]
                    labels_sel = labels_v[sel_idx]
                    loss = loss_fn(logits_sel, labels_sel)
                    batch_loss = loss if batch_loss is None else batch_loss + loss
                    valid_grad = True
                    n_loss_windows += 1
                    logits_all.append(logits_v.detach())          # stay on GPU
                    labels_all.append(labels_v.detach())

        elif not train_mode:
            logits = predict_edges(model, head, data)
            labels = data.label_t
            valid = labels != -1        # -1 = excluded edge (late-legs protocol)
            if valid.any():
                logits_v = logits[valid]
                labels_v = labels[valid]
                if loss_fn is not None:
                    loss = loss_fn(logits_v, labels_v)
                    total_loss += loss.item()
                    n_loss_windows += 1
                logits_all.append(logits_v.detach())
                labels_all.append(labels_v.detach())

        model.zoh_update(data)

        if train_mode and window_idx < t_warm:
            model.detach()

        if train_mode and window_idx >= t_warm:
            batch_counter += 1

        if train_mode and valid_grad and (batch_counter >= bptt_windows or data.end_train or window_idx == len(counter) - 1):
            accum_count = max(batch_counter, 1)
            batch_counter = 0
            optimizer.zero_grad()
            (batch_loss / accum_count).backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), max_norm=1.0)
            optimizer.step()
            model.detach()
            total_loss += batch_loss.item()
            batch_loss = None
            valid_grad = False

    if logits_all:
        logits = torch.cat(logits_all).cpu().numpy()          # single H2D-free transfer
        labels = torch.cat(labels_all).cpu().numpy().astype(np.int8)
        if log_odds_correction and (neg_sample_rate < 1.0):
            logits = logits + np.log(neg_sample_rate)
        metrics = compute_classification_metrics(
            labels, logits, eval_threshold=eval_threshold)
    else:
        logits = np.zeros(0)
        metrics = compute_classification_metrics(
            np.zeros(0, dtype=np.int8), logits, eval_threshold=eval_threshold)

    # Fix 4: Correct average window loss
    loss_out = total_loss / max(n_loss_windows, 1) if n_loss_windows > 0 else 0.0
    return loss_out, metrics


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across random, numpy, and torch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_edge_training(args):
    set_seed(getattr(args, 'seed', 42))
    t0 = time.time()
    print("[1/5] loading data ...")
    node_raw_features, edge_raw_features, full_data, num_nodes = load_all_edges(
        args.dataset, args.label_source, args.late_legs)
    edge_raw_features, tmx = prepare_features(args, node_raw_features, edge_raw_features, full_data)
    edge_feat_dim = edge_raw_features.shape[1]

    ts = full_data.node_interact_times
    assert np.all(np.diff(ts) >= 0), "expected chronologically ordered edges"
    labels = np.asarray(full_data.labels).astype(np.int8)   # -1 = excluded (late protocol)

    print("[2/5] Multi-GNN day-boundary split ...")
    train_mask, val_mask, test_mask, cuts = multiggn_day_split(
        ts.astype(np.float64), args.val_ratio, args.test_ratio)
    cut_i, cut_j = cuts
    print(f"    day-cuts at day {cut_i} and day {cut_j} (of "
          f"{int((ts.max()-ts.min())//86400)+1})")

    def _report(name, m):
        n = int(m.sum()); p = int((labels[m] == 1).sum()); x = int((labels[m] == -1).sum())
        print(f"    {name:5s} edges {n:10d} | illicit {p:6d} ({p/max(n,1)*100:.4f}%)"
              + (f" | excluded early-legs {x}" if x else ""))
    _report('train', train_mask); _report('val', val_mask); _report('test', test_mask)

    tr = dict(src=full_data.src_node_ids[train_mask], dst=full_data.dst_node_ids[train_mask],
              ts=ts[train_mask], ids=full_data.edge_ids[train_mask], labels=labels[train_mask])
    vl = dict(src=full_data.src_node_ids[val_mask], dst=full_data.dst_node_ids[val_mask],
              ts=ts[val_mask], ids=full_data.edge_ids[val_mask], labels=labels[val_mask])
    te = dict(src=full_data.src_node_ids[test_mask], dst=full_data.dst_node_ids[test_mask],
              ts=ts[test_mask], ids=full_data.edge_ids[test_mask], labels=labels[test_mask])

    # Class weight uses only labeled examples in the TRAIN split; excluded
    # early-legs (-1) count toward neither class.
    n_pos_tr = int((tr['labels'] == 1).sum())
    n_neg_tr = int((tr['labels'] == 0).sum())
    pos_weight_arg = str(args.pos_weight).strip().lower()
    neg_rate = float(getattr(args, 'neg_sample_rate', 1.0))
    raw_imbalance_ratio = n_neg_tr / max(n_pos_tr, 1)
    effective_imbalance_ratio = (n_neg_tr * neg_rate) / max(n_pos_tr, 1)
    if pos_weight_arg == 'auto':
        pw = effective_imbalance_ratio
        weight_description = f'auto effective (n_neg * neg_sample_rate / n_pos = {pw:.2f})'
    elif pos_weight_arg == 'sqrt':
        pw = np.sqrt(effective_imbalance_ratio)
        weight_description = f'sqrt effective (sqrt(n_neg * neg_sample_rate / n_pos) = {pw:.2f})'
    else:
        try:
            pw = float(pos_weight_arg)
        except ValueError as exc:
            raise ValueError("--pos_weight must be 'auto', 'sqrt', or a positive number") from exc
        if not np.isfinite(pw) or pw <= 0:
            raise ValueError("--pos_weight must be finite and greater than 0")
        weight_description = f'fixed ({pw:g})'
    print(f"[3/5] pos_weight = {pw:.4g} [{weight_description}] "
          f"(train pos {n_pos_tr}, neg {n_neg_tr}, neg_sample_rate {neg_rate:.2f})")
    loss_arg = str(getattr(args, 'loss', 'focal')).strip().lower()
    if loss_arg == 'focal':
        loss_fn = BinaryFocalLoss(gamma=args.gamma, pos_weight=pw)
        print(f"[3/5] loss = BinaryFocalLoss(gamma={args.gamma:g}, pos_weight={pw:.4g})")
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pw], dtype=torch.float, device=args.device))
        print(f"[3/5] loss = BCEWithLogitsLoss(pos_weight={pw:.4g})")

    results_summary = []
    use_wandb = bool(getattr(args, 'wandb', False))
    wandb_run = None
    if use_wandb:
        import wandb
        load_env()
        api_key = os.environ.get('WANDB_API_KEY')
        if not api_key:
            print("[wandb] WANDB_API_KEY not found in .env — continuing without wandb")
            use_wandb = False
    else:
        import wandb
    for run in range(args.runs):
        print(f"\n===== RUN {run+1}/{args.runs} =====")
        model, head = edge_model_with_head(args, num_nodes, edge_feat_dim,
                                           pw, n_pos_tr, n_neg_tr)
        wandb_run = None
        if use_wandb:
            try:
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    name=f"{args.name}{'_smoke' if args.smoke else ''}_run{run+1}",
                    group=args.name or 'EC',
                    job_type='edge_classification',
                    config=vars(args),
                    tags=[args.dataset, args.label_source],
                )
            except Exception as e:
                print(f"[wandb] init failed ({e}) — continuing without wandb")
                wandb_run = None

        print("[4/5] building datasets (train / val / test streams) ...")
        datasets = {}
        datasets['train'] = TemporalEdgeDataset(
            tr['src'], tr['dst'], tr['ts'], edge_raw_features[tr['ids']],
            biparted=False, T=args.batches, num_node=num_nodes,
            LastInteraction=tmx[tr['ids']], context_window=args.context,
            mode='train', device=args.device, task='NC', labels=tr['labels'],
            unique_batch=args.u_batch, negative='RNS')
        # eval datasets: context CSR = stream prefix (Multi-GNN: val graph
        # includes train; test graph = everything). The tau time-filter keeps
        # neighbor sampling causal regardless.
        c_src = np.hstack((tr['src'], vl['src']))
        c_dst = np.hstack((tr['dst'], vl['dst']))
        c_time = np.hstack((tr['ts'], vl['ts']))
        datasets['val'] = TemporalEdgeDataset(
            vl['src'], vl['dst'], vl['ts'], edge_raw_features[vl['ids']],
            biparted=False, T=args.batches, num_node=num_nodes,
            LastInteraction=tmx[vl['ids']], context_window=args.context,
            mode='eval', device=args.device, task='NC', labels=vl['labels'],
            unique_batch=args.u_batch, negative='RNS',
            c_src=c_src, c_dst=c_dst, c_time=c_time,
            inductive_mask=np.zeros(len(vl['src']), dtype=bool))
        c_src = np.hstack((c_src, te['src']))
        c_dst = np.hstack((c_dst, te['dst']))
        c_time = np.hstack((c_time, te['ts']))
        datasets['test'] = TemporalEdgeDataset(
            te['src'], te['dst'], te['ts'], edge_raw_features[te['ids']],
            biparted=False, T=args.batches, num_node=num_nodes,
            LastInteraction=tmx[te['ids']], context_window=args.context,
            mode='eval', device=args.device, task='NC', labels=te['labels'],
            unique_batch=args.u_batch, negative='RNS',
            c_src=c_src, c_dst=c_dst, c_time=c_time,
            inductive_mask=np.zeros(len(te['src']), dtype=bool))

        optimizer = optim.Adam(list(model.parameters()) + list(head.parameters()),
                               lr=args.lr, weight_decay=args.wd)

        # LR scheduler: monotonic cosine annealing (default), plateau, or none
        if args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
            scheduler_mode = 'epoch'  # step per epoch
        elif args.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.75, patience=args.lr_patience, min_lr=args.lr * 0.01)
            scheduler_mode = 'metric'
        else:
            scheduler = None
            scheduler_mode = 'none'

        best_val_metric, best_state, best_val_thr_prob = -1.0, None, 0.5
        patience_left = args.patience
        for epoch in range(1, args.epochs + 1):
            print(f"-- epoch {epoch}")
            torch.cuda.empty_cache()
            model.reset()
            tr_loss, tr_m = edge_epoch_stream(
                model, head, datasets['train'], optimizer, loss_fn,
                t_warm=args.t_warm, smoke=args.smoke,
                bptt_windows=args.bptt_windows,
                neg_sample_rate=args.neg_sample_rate,
                log_odds_correction=args.log_odds_correction,
                desc=f'Run{run+1} E{epoch} train')
            print(f"  epoch {epoch}: train loss {tr_loss:.4f} | F1@0.5 {tr_m['F1@0.5']:.4f} "
                  f"| F1_opt {tr_m['F1_opt']:.4f} (thr_prob {tr_m['best_thr_prob']:.4f}) "
                  f"| AP {tr_m['AP']:.4f} | AUC {tr_m['AUC']:.4f}")
            with torch.no_grad():
                val_loss, val_m = edge_epoch_stream(
                    model, head, datasets['val'], loss_fn=loss_fn,
                    smoke=args.smoke,
                    neg_sample_rate=args.neg_sample_rate,
                    log_odds_correction=args.log_odds_correction,
                    desc=f'Run{run+1} E{epoch} val')
            print(f"  epoch {epoch}: VAL   loss {val_loss:.4f} | F1@0.5 {val_m['F1@0.5']:.4f} "
                  f"| F1_opt {val_m['F1_opt']:.4f} (thr_prob {val_m['best_thr_prob']:.4f}) "
                  f"| AP {val_m['AP']:.4f} | AUC {val_m['AUC']:.4f}")
            if wandb_run is not None:
                wandb_run.log({
                    'epoch': epoch,
                    'train/loss': tr_loss,
                    'train/F1@0.5': tr_m['F1@0.5'],
                    'train/F1_opt': tr_m['F1_opt'],
                    'train/F1': tr_m['F1_opt'],
                    'train/AP': tr_m['AP'],
                    'train/AUC': tr_m['AUC'],
                    'val/loss': val_loss,
                    'val/F1@0.5': val_m['F1@0.5'],
                    'val/F1_opt': val_m['F1_opt'],
                    'val/F1': val_m['F1_opt'],
                    'val/AP': val_m['AP'],
                    'val/AUC': val_m['AUC'],
                    'val/best_thr_prob': val_m['best_thr_prob'],
                    'lr': optimizer.param_groups[0]['lr']
                })
            monitored_metric = val_m.get(args.eval_metric, val_m['F1_opt'])
            if scheduler is not None:
                if scheduler_mode == 'epoch':
                    scheduler.step()
                elif scheduler_mode == 'metric':
                    scheduler.step(monitored_metric)
            if monitored_metric > best_val_metric:
                best_val_metric = monitored_metric
                best_val_thr_prob = val_m['best_thr_prob']
                best_state = deepcopy({'model': model.state_dict(),
                                       'head': head.state_dict(),
                                       'best_thr': val_m['best_thr'],
                                       'best_thr_prob': best_val_thr_prob,
                                       'epoch': epoch,
                                       'metric_name': args.eval_metric,
                                       'metric_val': best_val_metric})
                patience_left = args.patience
                ckpt_path = f"{args.name}_run{run+1}_best_model.pt" if args.runs > 1 else f"{args.name}_best_model.pt"
                try:
                    torch.save(best_state, ckpt_path)
                    print(f"  --> Saved new best checkpoint to {ckpt_path} ({args.eval_metric}: {best_val_metric:.4f})")
                except Exception as e:
                    print(f"  --> Warning: could not save checkpoint to {ckpt_path}: {e}")
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"  early stopping (patience exhausted on val {args.eval_metric})")
                    break

        # ----- final test: reload best, burn-in over train+val, then classify -----
        if best_state is not None:
            model.load_state_dict(best_state['model'])
            head.load_state_dict(best_state['head'])
            best_val_thr_prob = best_state.get('best_thr_prob', 0.5)
            best_ep = best_state.get('epoch', 1)
            best_val_thr_logit = best_state.get('best_thr', 0.0)  # for logging
            print(f"  Reloaded best model from Epoch {best_ep} ({args.eval_metric}={best_val_metric:.4f}, thr_prob={best_val_thr_prob:.4f})")
        with torch.no_grad():
            model.eval()
            head.eval()
            model.reset()
            # Fix 3: full burn-in through the whole train and val stream
            if not args.no_burnin:
                tr_ds = datasets['train']
                if getattr(tr_ds, 'unique_batch', False):
                    tr_stream = range(len(tr_ds))
                    if args.smoke is not None:
                        tr_stream = range(min(len(tr_ds), args.smoke))
                else:
                    tr_stream = range(0, len(tr_ds), tr_ds.T)
                    if args.smoke is not None:
                        tr_stream = range(0, min(tr_stream.stop, args.smoke * tr_ds.T), tr_ds.T)
                for t in tqdm(tr_stream, desc='test burn-in (train)'):
                    model.zoh_update(tr_ds[t])

                vl_ds = datasets['val']
                if getattr(vl_ds, 'unique_batch', False):
                    vl_stream = range(len(vl_ds))
                    if args.smoke is not None:
                        vl_stream = range(min(len(vl_ds), args.smoke))
                else:
                    vl_stream = range(0, len(vl_ds), vl_ds.T)
                    if args.smoke is not None:
                        vl_stream = range(0, min(vl_stream.stop, args.smoke * vl_ds.T), vl_ds.T)
                for t in tqdm(vl_stream, desc='test burn-in (val)'):
                    model.zoh_update(vl_ds[t])
            te_loss, te_m = edge_epoch_stream(
                model, head, datasets['test'], loss_fn=loss_fn,
                eval_threshold=best_val_thr_prob, smoke=args.smoke,
                neg_sample_rate=args.neg_sample_rate,
                log_odds_correction=args.log_odds_correction,
                desc='test')
        print(f"  TEST  F1@0.5 {te_m['F1@0.5']:.4f} "
              f"| F1_opt(val_thr_prob={best_val_thr_prob:.4f}) {te_m['F1_opt']:.4f} "
              f"| AP {te_m['AP']:.4f} | AUC {te_m['AUC']:.4f}")
        if wandb_run is not None:
            try:
                wandb_run.log({
                    'test/loss': te_loss,
                    'test/F1@0.5': te_m['F1@0.5'],
                    'test/F1_opt': te_m['F1_opt'],
                    'test/F1': te_m['F1_opt'],
                    'test/AP': te_m['AP'],
                    'test/AUC': te_m['AUC'],
                    'best_val_metric': best_val_metric,
                    'best_val_thr_prob': best_val_thr_prob,
                })
                wandb_run.finish()
            except Exception as e:
                print(f"[wandb] upload failed: {e}")
            wandb_run = None
        results_summary.append({'val': best_val_metric, 'test': te_m})
        del datasets, model, head
        torch.cuda.empty_cache(); gc.collect()

    # ----- aggregate across runs -----
    print("\n============= Edge classification summary over runs =============")
    f1_05s = [r['test'].get('F1@0.5', float('nan')) for r in results_summary]
    f1_opts = [r['test'].get('F1_opt', float('nan')) for r in results_summary]
    aps = [r['test'].get('AP', float('nan')) for r in results_summary]
    aucs = [r['test'].get('AUC', float('nan')) for r in results_summary]

    def ms(x):
        x = np.array(x, dtype=np.float64)
        return f"{x.mean():.4f} ± {x.std():.4f}"
    print(f"  F1@0.5 (thr 0.5): {ms(f1_05s)}")
    print(f"  F1_opt (val thr): {ms(f1_opts)}")
    print(f"  AP     (PR-AUC) : {ms(aps)}")
    print(f"  AUC             : {ms(aucs)}")
    print(f"[total wall time {time.time()-t0:.1f}s]")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str(v).lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Temporal Edge (AML) Classification")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="li_small")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Multi-GNN-style day split ratio (train gets the rest)")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--time_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--embd_dim", type=int, default=16)
    parser.add_argument("--context", type=int, default=10)
    parser.add_argument("--reg", type=float, default=1e-2)
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=2.0e-4,
                        help="learning rate (default 2.0e-4 for stable continuous SSM training)")
    parser.add_argument("--wd", type=float, default=1e-4,
                        help="weight decay (default 1e-4)")
    parser.add_argument("--epochs", type=int, default=15,
                        help="max epochs (default 15 for cosine schedule)")
    parser.add_argument("--batches", type=int, default=256)
    parser.add_argument("--u_batch", type=str2bool, default=False,
                        help="ensure no node repeats in a batch")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--t_warm", type=int, default=100,
                        help="memory-only warm-up windows per train epoch (LP convention)")
    parser.add_argument("--head_type", type=str, default="cross",
                        choices=["linear", "mlp", "cross", "dcn"],
                        help="edge prediction head architecture: 'cross' (DCN explicit feature crossing, default), "
                             "'linear' (LayerNorm + Linear), or 'mlp' (LayerNorm + GELU MLP)")
    parser.add_argument("--bptt_windows", type=int, default=25,
                        help="number of consecutive windows before gradient step & model.detach() "
                             "(default 25; captures ~6,400 edges / ~25 minutes of continuous graph dynamics)")
    parser.add_argument("--neg_sample_rate", type=float, default=0.15,
                        help="fraction of negative transactions to sample for training loss (default 0.15 = 15 pct). "
                             "100 pct of positives are ALWAYS kept for loss. 100 pct of all transactions update Mamba memory "
                             "(zoh_update). 100 pct of validation/test transactions are evaluated without subsampling. "
                             "Set to 1.0 to disable negative subsampling.")
    parser.add_argument("--log_odds_correction", type=str2bool, default=False,
                        help="apply Bayes log-odds correction (logit + log(neg_sample_rate)) during evaluation "
                             "when negative subsampling is used, aligning raw probability threshold 0.5 with true prior")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for torch/numpy reproducibility (default 42)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="dropout rate in prediction head (default 0.1)")
    parser.add_argument("--prune_features", type=str2bool, default=True,
                        help="prune 139 zero columns from pre-baked IBM-AML features "
                             "(reduces input dim from 172 to 33, eliminating dummy weights)")
    parser.add_argument("--use_relational", type=str2bool, default=True,
                        help="augment prediction head with bilinear relational interactions "
                             "(hadamard product and absolute difference of node states)")
    parser.add_argument("--loss", type=str, default="focal",
                        choices=["bce", "focal"],
                        help="loss function: 'focal' (Focal Loss downweighting easy negatives) or 'bce'")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="focusing parameter gamma for Focal Loss (default 2.0)")
    parser.add_argument("--patience", type=int, default=6,
                        help="early stopping patience (epochs without validation improvement)")
    parser.add_argument("--lr_patience", type=int, default=3,
                        help="ReduceLROnPlateau patience (epochs before LR reduction)")
    parser.add_argument("--label_source", type=str, default="flag",
                        choices=["flag", "patterns", "late"])
    parser.add_argument("--late_legs", type=int, default=5,
                        help="label_source=late: legs within the last K positions of "
                             "their attempt get label 1; earlier legs excluded")
    parser.add_argument("--pos_weight", type=str, default="6.0",
                        help="positive-class weight: '6.0' (default, calibrated with focal loss), "
                             "'auto' = effective n_neg/n_pos, 'sqrt' = sqrt(effective n_neg/n_pos), or any positive number")
    parser.add_argument("--smoke", type=int, default=None,
                        help="run only the first N windows of each pass")
    parser.add_argument("--no_burnin", action="store_true")
    parser.add_argument("--eval_metric", type=str, default="AP",
                        choices=["AP", "AUC", "F1_opt", "F1@0.5"],
                        help="validation metric to monitor for LR schedule & early stopping")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau", "none"],
                        help="LR scheduler: 'cosine' (monotonic cosine annealing, default), 'plateau', or 'none'")
    parser.add_argument("--cosine_t0", type=int, default=10,
                        help="CosineAnnealingWarmRestarts T_0 (initial restart period)")
    parser.add_argument("--cosine_tmult", type=int, default=2,
                        help="CosineAnnealingWarmRestarts T_mult (multiplier for restart period)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--name", type=str, default="EC_")
    parser.add_argument("--wandb", action="store_true",
                        help="log metrics to Weights & Biases (WANDB_API_KEY in .env)")
    parser.add_argument("--wandb_project", type=str, default="ctdg-ssm-ec")
    args = parser.parse_args()

    if args.config is not None:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k) and v is not None:
                setattr(args, k, v)

    # type enforcement (argparse + config merge)
    for k, t in (('lr', float), ('wd', float), ('val_ratio', float), ('test_ratio', float),
                 ('time_dim', int), ('hidden_dim', int), ('embd_dim', int),
                 ('batches', int), ('context', int), ('epochs', int), ('runs', int),
                 ('reg', float), ('delta', float), ('u_batch', str2bool), ('patience', int),
                 ('lr_patience', int), ('head_type', str), ('use_relational', str2bool),
                 ('prune_features', str2bool), ('neg_sample_rate', float),
                 ('log_odds_correction', str2bool), ('seed', int), ('dropout', float),
                 ('loss', str), ('gamma', float), ('bptt_windows', int),
                 ('t_warm', int), ('late_legs', int), ('wandb', bool), ('eval_metric', str),
                 ('scheduler', str), ('cosine_t0', int), ('cosine_tmult', int)):
        if hasattr(args, k):
            setattr(args, k, t(getattr(args, k)))
    if args.smoke is not None:
        args.smoke = int(args.smoke)

    print("Arguments used:")
    for a, v in vars(args).items():
        print(f"  {a}: {v}")
    run_edge_training(args)


if __name__ == "__main__":
    main()
