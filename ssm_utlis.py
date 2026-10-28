
import torch
import numpy as np
import pandas as pd
import random
import copy

def print_eval_metrics(results, prefix=""):
    """
    results: dict returned from link_prediction_eval
    prefix: string to prepend (e.g. 'VAL' or 'TEST')
    """
    for split in ['transductive', 'inductive']:
        print(f"\n{prefix} {split.upper()} metrics:")
        for neg_type in ['rnd', 'hst', 'ind']:
            auc = results[split][neg_type]['AUC']
            ap = results[split][neg_type]['AP']
            print(f"  {neg_type.upper():>5} - AUC: {auc:.4f} | AP: {ap:.4f}")

def aggregate_metrics(results_list, split_name):
    # results_list is list of dicts: each dict like {'rnd': {'AUC':.., 'AP':..}, ...}
    metrics = {'rnd': {'AUC': [], 'AP': []}, 'hst': {'AUC': [], 'AP': []}, 'ind': {'AUC': [], 'AP': []}}

    for res in results_list:
        for neg_type in ['rnd', 'hst', 'ind']:
            metrics[neg_type]['AUC'].append(res[neg_type]['AUC'])
            metrics[neg_type]['AP'].append(res[neg_type]['AP'])

    # Compute mean ± std per neg_type and metric
    summary = {}
    for neg_type in metrics:
        summary[neg_type] = {
            'AUC_mean': np.mean(metrics[neg_type]['AUC']),
            'AUC_std': np.std(metrics[neg_type]['AUC']),
            'AP_mean': np.mean(metrics[neg_type]['AP']),
            'AP_std': np.std(metrics[neg_type]['AP']),
        }
    return summary


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0.001, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as improvement.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.best_state_dict = None

    def __call__(self, val_loss, model):
        score = val_loss  # lower loss = higher score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score* 1 + self.delta :
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def combine_sort(dataset, new_dataset):
    """
    Merge old and new edge lists, sort by time,
    and return (src, dst, time, mask).
    
    mask[i] = 0 if edge came from old set,
              1 if edge came from new set
    """

    combined_dataset = copy.deepcopy(dataset)

    # Masks
    old_mask = np.zeros_like(dataset.node_interact_times, dtype=bool)  # 0 = old
    new_mask = np.ones_like(new_dataset.node_interact_times, dtype=bool)   # 1 = new

    # Concatenate
    src = np.concatenate([dataset.src_node_ids, new_dataset.src_node_ids])
    dst = np.concatenate([dataset.dst_node_ids, new_dataset.dst_node_ids])
    time = np.concatenate([dataset.node_interact_times, new_dataset.node_interact_times])
    mask = np.concatenate([old_mask, new_mask])
    edge_ids = np.concatenate([dataset.edge_ids, new_dataset.edge_ids])

    # Sort by time
    idx = np.argsort(time)

    combined_dataset.src_node_ids = src[idx]
    combined_dataset.dst_node_ids = dst[idx]
    combined_dataset.node_interact_times = time[idx]
    combined_dataset.inductive_mask = mask[idx]
    combined_dataset.edge_ids= edge_ids[idx]
    return combined_dataset
    

def build_laplacian_windowed_agg(src, dst, num_nodes,weight, device):
    if weight== None:
        weight = torch.ones(src.size(0), device=src.device)
    # FIX (speedup #3 in INFERENCE_SPEEDUP_PLAN.md §5): replace
    # stack + sparse_coo_tensor + to_dense with a direct dense scatter. Bit-exact:
    # weights are 1.0, so duplicate-key sums are exact fp32 integers, and A is
    # binarized next line anyway → identical final matrix. Saves ~6 kernels
    # per window (stack + coo + densify). (Only caller passes 1-D src/dst,
    # weight=None — verified in-repo.)
    A = torch.zeros(num_nodes, num_nodes, device=device)
    A.index_put_((src, dst), weight, accumulate=True)

    # symmetrize and binarize adjacency
    A = ((A + A.T) > 0).float()

    # degree inverse
    D = torch.diag(torch.clamp_min(A.sum(dim=1),1)**(-0.5))

    # Laplacian: L = I - D^{-1} A
    L = torch.eye(num_nodes, device=device) - D@A@D
    return L

def build_laplacian_windowed(tau,T,src,dst,src_t,dst_t,active_ids,sub_src, sub_dst, weight=None, device='cpu'):
    if weight is None:
        weight = torch.ones(src.size(0), device=device)

    # Build adjacency matrix (dense once)
    # edge_index = torch.stack([sub_src, sub_dst], dim=0)
    # A = torch.sparse_coo_tensor(edge_index, weight, (num_nodes, num_nodes), device=device).to_dense()

    
    # # Degree vector
    # D = torch.sum(A, dim=1)
    # D[D == 0] = 1.0
    # D_inv = 1.0 / D

    # Normalized Laplacian: L = I - D^-1 * A
    # Multiply each row of A by D_inv
    L = torch.eye(len(active_ids), device=device) 

    src_tau = src[:tau]
    dst_tau = dst[:tau]
    src_taup = src[:tau+T]
    dst_taup = dst[:tau+T]
    # Delta_L:
    delta_L = torch.zeros_like(L,device=device)
    for i,node in enumerate(active_ids):
        new_src = src_t[dst_t==node]
        if len(new_src)==0:
            continue
        new_src_ids = sub_src[dst_t==node]
        old_nu = src_tau[dst_tau==node]
        du = old_nu.numel()
        if du == 0:
            du = 1

        new_nu = src_taup[dst_taup==node]
        du_ = new_nu.numel()
        if du_ ==0:
            du_ = 1

        old_w = torch.hstack([torch.sum(old_nu==src_x)/du for src_x in new_src]) 
        new_w = torch.hstack([torch.sum(new_nu==src_x)/du_ for src_x in new_src]) 
        delta_L[i,new_src_ids] = old_w-new_w
        L[i,new_src_ids] = -new_w
    return L, delta_L

def positional_encoding_mixer(timestamps, num_dims):
    alpha = num_dims**(0.5)
    beta  = num_dims**(0.5)
    i = np.arange(num_dims)[None]
    omega_raise = alpha**((-i+1)/beta)
    pos = timestamps[:,None]  # [T, 1]
    
    return np.cos(pos * omega_raise)

# def aggregate_edge_features_to_nodes(src, dst, edge_feats, num_nodes):
#     x = torch.zeros(num_nodes, edge_feats.shape[1], device=edge_feats.device)
#     for i in range(len(edge_feats)):
#         x[src[i]] += edge_feats[i]
#         x[dst[i]] += edge_feats[i]
#     return x

# def aggregate_edge_features_to_nodes(src, dst, edge_feats, num_nodes):
#     x = torch.zeros(num_nodes, edge_feats.shape[1], device=edge_feats.device)
#     x.index_add_(0, src, edge_feats)  # add edge features to source nodes
#     x.index_add_(0, dst, edge_feats)  # add edge features to destination nodes
#     return x

def aggregate_edge_features_to_nodes(src, dst, edge_feats, num_nodes):
    """
    Vectorized aggregation of edge features to nodes, with zero-padding if edge_feats is shorter than src/dst.
    Args:
        src (LongTensor): source node indices of edges, shape [num_edges_total]
        dst (LongTensor): destination node indices of edges, shape [num_edges_total]
        edge_feats (FloatTensor): edge features, shape [num_edges_original, feat_dim]
        num_nodes (int): number of nodes in graph

    Returns:
        x (FloatTensor): node features aggregated from edges, shape [num_nodes, feat_dim]
    """
    num_edges_total = len(src)
    feat_dim = edge_feats.shape[1]

    # Pad edge_feats with zeros if there are missing edges
    if edge_feats.shape[0] < num_edges_total:
        pad_len = num_edges_total - edge_feats.shape[0]
        pad = torch.zeros(pad_len, feat_dim, device=edge_feats.device, dtype=edge_feats.dtype)
        edge_feats = torch.cat([edge_feats, pad], dim=0)

    # Preallocate node features
    x = torch.zeros(num_nodes, feat_dim, device=edge_feats.device, dtype=edge_feats.dtype)

    # Vectorized addition
    x.index_add_(0, src, edge_feats)
    x.index_add_(0, dst, edge_feats)

    return x

def dir_causal_laplacain(src,dst,delta=1,device='cpu'):
    num_nodes = len(src)*2
    num_src = len(src)
    A = torch.zeros((num_nodes,num_nodes)).to(device)
    for t,node in enumerate(src):
        nindex = num_src+torch.arange(len(src[:t+1]))
        n_mask = nindex[src[:t+1]==node]
        A[t,n_mask] = torch.ones_like(n_mask,device=device,dtype=torch.float32) #torch.exp(-(t-src_mask)/delta)

    for t,node in enumerate(dst):
        #sindex = torch.arange(len(dst[:t+1]))
        nindex = torch.arange(len(dst[:t+1]))
        n_mask = nindex[dst[:t+1]==node]
        A[t+num_src,n_mask] = torch.ones_like(n_mask,device=device,dtype=torch.float32) #torch.exp(-(t-src_mask)/delta)


    A += torch.eye(len(A)).to(device)
    D = torch.sum(A,dim=1)[:,None]
    L = torch.eye(len(A)).to(device)-(1/D)*A

    return L

# def node_unique_batch(src, dst, T, t_start):
#     """
#     src, dst : LongTensor [num_edges]
#     T        : max window size
#     t_start  : current start index
#     """
#     # Take candidate window of size T
#     end_idx = min(t_start + T, src.size(0))
#     src_win = src[t_start:end_idx]
#     dst_win = dst[t_start:end_idx]

#     # Stack edges into node sequence [2, window_size]
#     nodes = torch.cat([src_win, dst_win])

#     # Find first repeat
#     # index of first duplicate == cutoff point
#     _, inv, counts = torch.unique(nodes, return_inverse=True, return_counts=True)
#     repeat_mask = counts[inv] > 1
#     if repeat_mask.any():
#         # position of first repeat in nodes
#         cutoff = repeat_mask.nonzero(as_tuple=False)[0, 0] // 2
#         cutoff = max(1, cutoff)  

#         end_idx = t_start + cutoff
#         src_win = src[t_start:end_idx]
#         dst_win = dst[t_start:end_idx]

#     return src_win, dst_win, end_idx

def node_unique_batch(src, dst, T, t_start):
    """
    src, dst : LongTensor [num_edges]
    T        : max window size
    t_start  : current start index
    """
    # Take candidate window of size T
    end_idx = min(t_start + T, src.size(0))
    src_win = src[t_start:end_idx]
    dst_win = dst[t_start:end_idx]

    # Track seen nodes incrementally (set-based, fast)
    seen = set()
    cutoff = None
    for i, (u, v) in enumerate(zip(src_win.tolist(), dst_win.tolist())):
        if u in seen or v in seen:
            cutoff = i  # first repeat-causing edge
            break
        seen.add(u)
        seen.add(v)

    if cutoff is not None:
        end_idx = t_start + cutoff
        src_win = src[t_start:end_idx]
        dst_win = dst[t_start:end_idx]

    return src_win, dst_win, end_idx

def sample_uniform_fast(nodes, past_self, past_neighbour, k,mode='rnd'):
    node_neighbours = []
    node_neighbours_nodes = []
    
    for node in nodes:
        neighbors = past_neighbour[past_self == node]
        n = neighbors.numel()
        
        if n == 0:
            continue  # no neighbors, just skip
        
        if n <= k:
            sampled = neighbors
        else:
            if mode == 'rnd':
                idx = torch.randint(0, n, (k,), device=neighbors.device)
                sampled = neighbors[idx]
            elif mode== 'rcn':
                sampled = neighbors[:k]
        
        node_neighbours.append(sampled)
        node_neighbours_nodes.append(node.repeat(sampled.numel()))
    
    if len(node_neighbours) == 0:
        return torch.empty(0, dtype=torch.long, device=past_self.device), \
               torch.empty(0, dtype=torch.long, device=past_self.device)
    
    return torch.hstack(node_neighbours), torch.hstack(node_neighbours_nodes)


def NeighbourSampler(src_t,dst_t,neg_dst,src,dst,t_start,tau=1,max_node=20,mode='CW'):
    if mode=='CW':
        if t_start>tau:
            dst_tau = dst[t_start-tau:t_start]
            src_tau = src[t_start-tau:t_start]
        else : 
            dst_tau = dst[:t_start]
            src_tau = src[:t_start]

        src_neighbours,src_neighbours_src  = sample_uniform_fast(src_t,src_tau,dst_tau,max_node) 
        dst_neighbours,dst_neighbours_dst =sample_uniform_fast(dst_t,dst_tau,src_tau,max_node) 
        neg_neighbour,neg_neighbour_dst = sample_uniform_fast(neg_dst,dst_tau,src_tau,max_node)
        
    elif mode=='FX':
        dst_tau = dst[:t_start]
        src_tau = src[:t_start]

        src_neighbours,src_neighbours_src  = sample_uniform_fast(src_t,src_tau,dst_tau,max_node,'rcn') 
        dst_neighbours,dst_neighbours_dst =sample_uniform_fast(dst_t,dst_tau,src_tau,max_node,'rcn') 
        neg_neighbour,neg_neighbour_dst = sample_uniform_fast(neg_dst,dst_tau,src_tau,max_node,'rcn')
    
    return src_neighbours,src_neighbours_src,dst_neighbours,dst_neighbours_dst,neg_neighbour,neg_neighbour_dst


def add_edge_features(src,dst,t,edge_raw_features,time_encoder,time_enc_dim_data):
    """
    Memory-lean version of add_edge_features (behaviour identical to the original):
      - groups on an int64 pair key instead of 6.9M Python tuples
      - computes the time encoding in chunks and stores it as float32
      - assembles the final feature matrix as float32
        (torch consumes it as float32 anyway; f1 counts stay exact because 2**24 > 6.9M)
    Returns (final_edge_feat [float32, N x F], tmx [float64, unlogged f2 incl. first-occurrence fill])
    """
    src = np.asarray(src)
    dst = np.asarray(dst)
    t = np.asarray(t)
    N = len(t)

    # integer (src,dst) pair key; unique per pair, tiny memory footprint
    pair_code = src.astype(np.int64) * (int(dst.max()) + 1) + dst

    df = pd.DataFrame({'code': pair_code, 't': t})

    # f1: cumulative count of (u,v)
    f1 = df.groupby('code').cumcount().to_numpy()

    # f2: time since last occurrence
    last_time = df.groupby('code')['t'].shift().to_numpy()
    df = None
    del pair_code

    f2 = t - last_time
    del last_time
    f2[np.isnan(f2)] = 1e+11          # mark first occurrence
    # NOTE: the original mutated df['f2'] with log1p before returning it,
    # so tmx here is the LOGGED f2 (first occurrences become ~25.33), as before.
    f2 = np.log1p(f2)
    tmx = f2

    edge_raw_features = np.asarray(edge_raw_features)
    raw_dim = edge_raw_features.shape[1]
    feat_dim = raw_dim + time_enc_dim_data + 1
    final_edge_feat = np.empty((N, feat_dim), dtype=np.float32)
    final_edge_feat[:, :raw_dim] = edge_raw_features
    del edge_raw_features

    # time encoding in chunks so the float64 intermediate stays small
    chunk = 1_000_000
    for i in range(0, N, chunk):
        enc = time_encoder(f2[i:i+chunk], time_enc_dim_data)
        final_edge_feat[i:i+chunk, raw_dim:raw_dim+time_enc_dim_data] = enc.astype(np.float32)
        enc = None
    final_edge_feat[:, -1] = f1.astype(np.float32)
    return final_edge_feat, tmx


#%% Rapid Neighbour Sampling

# def build_neighbor_index_torch(df, num_nodes=None, device="cpu"):
#     """
#     Build neighbor adjacency in contiguous torch tensors with offsets.
#     Each node's neighbors are stored in a single tensor slice:
#         neighbors[offsets[i]:offsets[i+1]]
#         times[offsets[i]:offsets[i+1]]
    
#     Args:
#         df : pandas.DataFrame with columns [src, dst, t]
#         num_nodes : optional, number of nodes (if None → inferred)
#         device : "cpu" or "cuda"
    
#     Returns:
#         neighbors: torch.LongTensor [E_total]
#         times: torch.LongTensor [E_total]
#         offsets: torch.LongTensor [num_nodes+1]
#     """
#     # infer num_nodes
#     if num_nodes is None:
#         num_nodes = int(max(df["src"].max(), df["dst"].max())) + 1

#     # adjacency list as python first
#     adj = [[] for _ in range(num_nodes)]
#     for src, dst, t in df.itertuples(index=False):
#         adj[src].append((t, dst))
#         adj[dst].append((t, src))

#     # sort and flatten
#     neighbors_list, times_list, offsets = [], [], [0]
#     for node in range(num_nodes):
#         nbrs = sorted(adj[node], key=lambda x: x[0])  # sort by time
#         if nbrs:
#             tvals, nbrs_ids = zip(*nbrs)
#             neighbors_list.extend(nbrs_ids)
#             times_list.extend(tvals)
#         offsets.append(len(neighbors_list))

#     # convert to torch tensors
#     neighbors = torch.tensor(neighbors_list, dtype=torch.long, device=device)
#     times = torch.tensor(times_list, dtype=torch.long, device=device)
#     offsets = torch.tensor(offsets, dtype=torch.long, device=device)

#     return neighbors, times, offsets

import torch

def build_neighbor_index_torch(src, dst, t, num_nodes=None, device="cpu"):
    """
    Build neighbor adjacency in contiguous torch tensors with offsets.
    Assumes (src, dst, t) are already sorted by time.

    Args:
        src, dst : LongTensors [E]
        t        : LongTensor [E]
        num_nodes : optional, number of nodes (if None → inferred)
        device : "cpu" or "cuda"

    Returns:
        neighbors: LongTensor [2*E]
        times: LongTensor [2*E]
        offsets: LongTensor [num_nodes+1]
    """
    if num_nodes is None:
        num_nodes = int(torch.max(torch.cat([src, dst])).item()) + 1

    # undirected expansion
    neighbors = torch.cat([dst, src])
    times     = torch.cat([t, t])
    nodes     = torch.cat([src, dst])

    # FIX (2026-10-26, user-approved): sort by node so that
    # neighbors[offset[u]:offset[u+1]] is really node u's neighbor list
    # (previously offsets were bincounts over an UNSORTED time stream, so each
    # node's "segment" was an arbitrary slice of the edge stream). Stable sort
    # preserves chronological (time) order within each node, exactly like the
    # original commented-out python-loop builder above.
    order = torch.argsort(nodes, stable=True)
    neighbors = neighbors[order]
    times     = times[order]
    nodes     = nodes[order]

    # just count how many per node → offsets
    counts = torch.bincount(nodes, minlength=num_nodes)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long,device=device), counts.cumsum(0)])

    return neighbors.to(device), times.to(device), offsets.to(device)


def temporal_sample_neighbors(src_nodes, neighbors, times, offsets, tau, k, replace=False):
    """
    Vectorized temporal neighbor sampler — no Python loops, no device syncs
    (one .item() for the total size only).

    src_nodes: (B,) LongTensor, batch of nodes
    neighbors: (E,) LongTensor, all neighbors (CSR)
    times:     (E,) LongTensor, timestamps aligned with neighbors
    offsets:   (N+1,) LongTensor, CSR offsets
    tau:       int, cutoff timestamp
    k:         int, number of neighbors to sample
    replace:   bool, whether to sample with replacement

    Returns:
        nbrs:  (B, k) LongTensor (padded with -1)
    """
    device = neighbors.device
    B = src_nodes.numel()

    # get neighbor index ranges for each src
    start = offsets[src_nodes]
    end   = offsets[src_nodes + 1]
    counts = end - start  # number of neighbors per node

    # ---- flat candidate indices (was: torch.cat([arange(s,e) for s,e in zip(...)])
    # ---- which force-synced zero-copy device->host per element, twice)
    c_cumsum = counts.cumsum(0)
    seg_off  = c_cumsum - counts          # exclusive offsets in flat space
    total = int(c_cumsum[-1].item())      # ONE sync per call
    flat = (start.repeat_interleave(counts)
            + torch.arange(total, device=device)
            - seg_off.repeat_interleave(counts))

    # gather neighbor + time (single gather, order matches old implementation)
    all_nbrs  = neighbors[flat]
    all_times = times[flat]

    # map each flat idx to its src node
    node_ids = torch.repeat_interleave(torch.arange(B, device=device), counts)

    # filter by tau
    mask = all_times <= tau
    all_nbrs, all_times, node_ids = all_nbrs[mask], all_times[mask], node_ids[mask]

    # now group back by src node
    # build (B, k) outputs
    nbrs_out = torch.full((B, k), -1, dtype=torch.long, device=device)
    # tvals_out = torch.full((B, k), -1, dtype=torch.long, device=device)

    # sort neighbors by time (latest first for consistency)
    order = torch.argsort(all_times, descending=True, stable=True)
    all_nbrs, all_times, node_ids = all_nbrs[order], all_times[order], node_ids[order]

    # pack into output using segment fill
    idx_within = torch.zeros_like(node_ids)
    idx_within[1:] = (node_ids[1:] == node_ids[:-1]).cumsum(0)

    mask_k = idx_within < k
    all_nbrs, all_times, node_ids, idx_within = (
        all_nbrs[mask_k],
        all_times[mask_k],
        node_ids[mask_k],
        idx_within[mask_k],
    )

    nbrs_out[node_ids, idx_within] = all_nbrs
    # tvals_out[node_ids, idx_within] = all_times

    return nbrs_out #, tvals_out


def batch_indices_fast(src, dst, B, num_nodes=None):
    """
    Fast batching:
      - max batch size = B
      - no node repeats per batch
    Identical greedy logic to the original, compiled with numba over numpy
    arrays (the previous version did one `.item()` per edge on GPU tensors —
    a device sync per edge at dataset-build time).

    Args:
        src, dst: LongTensor (E,) — any device; copied to numpy once
        B: int, max batch size
        num_nodes: int (optional, max node id+1 for mask)
    Returns:
        start, end: LongTensor batch boundaries (cpu)
    """
    s = np.asarray(src.detach().to('cpu').numpy(), dtype=np.int64)
    d = np.asarray(dst.detach().to('cpu').numpy(), dtype=np.int64)
    E = len(s)
    if num_nodes is None:
        num_nodes = int(max(s.max(), d.max())) + 1

    starts, ends = (_batch_indices_numba(s, d, B, num_nodes)
                    if _NUMBA else _batch_indices_py(s, d, B, num_nodes))
    return torch.tensor(starts), torch.tensor(ends)


try:
    from numba import njit as _njit
    _NUMBA = True
except Exception:                       # numba missing -> pure-python fallback
    _njit = None
    _NUMBA = False


def _batch_indices_py(s, d, B, num_nodes):
    E = len(s)
    used = np.zeros(num_nodes, dtype=np.bool_)
    starts = np.empty(E + 1, dtype=np.int64)
    ends = np.empty(E + 1, dtype=np.int64)
    nb = 0
    batch_start = 0
    count = 0
    for i in range(E):
        u, v = s[i], d[i]
        if count >= B or used[u] or used[v]:
            starts[nb] = batch_start
            ends[nb] = i
            nb += 1
            batch_start = i
            used[:] = False
            count = 0
        used[u] = True
        used[v] = True
        count += 1
    starts[nb] = batch_start
    ends[nb] = E
    nb += 1
    return starts[:nb], ends[:nb]


if _NUMBA:
    @_njit(cache=True)
    def _batch_indices_numba(s, d, B, num_nodes):
        # 'stamp' replaces the boolean reset: stamp[u] == epoch  <=>  used[u]
        # (vectorized whole-array reset replaced by an O(1) epoch bump).
        # init to -1 so untouched nodes never match epoch 0 (that produced a
        # spurious empty first batch).
        E = s.shape[0]
        stamp = np.full(num_nodes, -1, dtype=np.int64)
        epoch = np.int64(0)
        starts = np.empty(E + 1, dtype=np.int64)
        ends = np.empty(E + 1, dtype=np.int64)
        nb = 0
        batch_start = 0
        count = 0
        for i in range(E):
            u, v = s[i], d[i]
            if count >= B or stamp[u] == epoch or stamp[v] == epoch:
                starts[nb] = batch_start
                ends[nb] = i
                nb += 1
                batch_start = i
                epoch = epoch + np.int64(1)
                count = 0
            stamp[u] = epoch
            stamp[v] = epoch
            count += 1
        starts[nb] = batch_start
        ends[nb] = E
        nb += 1
        return starts[:nb], ends[:nb]
else:
    def _batch_indices_numba(s, d, B, num_nodes):
        return _batch_indices_py(s, d, B, num_nodes)

#%% AGG

def scatter_mean_update(state_vec, idx, vals):
    """
    Perform gradient-preserving mean update on state_vec at indices idx.
    Handles repeated indices by averaging their updates.
    """
    # Sums across repeated indices
    sums = torch.zeros_like(state_vec).index_add(0, idx, vals)

    # Counts per index
    counts = torch.zeros(state_vec.size(0), device=vals.device).index_add(
        0, idx, torch.ones(idx.size(0), device=vals.device)
    )

    # Compute means (only where counts > 0)
    means = torch.where(
        counts[:, None] > 0, sums / counts[:, None], state_vec
    )

    # Update only active indices
    return torch.where(
        counts[:, None] > 0,
        means,
        state_vec
    )

def _build_pair_table(neighbors, times, offsets):
    """Undirected-pair index, sorted lexicographically by (pair_key, time_rank).

    Returns (lex, ltime, tsorted, R):
      lex    : int64   [2E]  composite = pair_key * R + time_rank, globally ascending
                       (time_rank = index of the entry's time in the globally
                        sorted time order). One searchsorted(lex, q*R + rank(t)
                        then answers "last interaction of this pair strictly
                        before t" EXACTLY, for arbitrary t — a plain
                        searchsorted over a key-sorted array cannot, because
                        entry times are not globally ascending across segments
                        (the earlier implementation did exactly that → garbage
                        j → negative delta → log1p → NaN; found by the LP NaN
                        tracer on li_small).
      ltime  : float32 [2E]  interaction time at each lex position
      tsorted: float32 [2E]  globally sorted entry times (for ranking queries)
      R      : int           modulus used in the composite == entries+1
    NOTE: composite can overflow int64 on very large graphs
    (max_key * (2E+1) >= 2^63); that raises here — LI-Large needs a two-level
    key or a numba bisect instead.
    """
    device = neighbors.device
    num_nodes = offsets.numel() - 1
    counts = offsets[1:] - offsets[:-1]
    node_rep = torch.repeat_interleave(torch.arange(num_nodes, device=device), counts)
    key = (torch.minimum(node_rep, neighbors).long() * num_nodes
           + torch.maximum(node_rep, neighbors).long())
    tflat = times.reshape(-1).to(device).float()

    tsorted = torch.sort(tflat).values
    R = int(tflat.numel()) + 1
    # time rank of every entry = number of entries with strictly smaller time
    rank = torch.searchsorted(tsorted, tflat, side='left')
    # ties share a rank; the strict '<' query-side comparison then excludes
    # interactions AT t exactly like the old loop's `times_u < t` mask.

    max_key = int(key.max())
    if max_key * R + R >= (1 << 63):
        raise OverflowError(
            "get_delta_t lexicographic composite overflows int64 for this "
            "graph size; needs a two-level key or numba bisect fallback")

    lex = (key * R + rank).contiguous()
    order = torch.argsort(lex)
    return lex[order].contiguous(), tflat[order].contiguous(), tsorted.contiguous(), R


_PAIR_CACHE = {}   # id(neighbors tensor) -> (lex, ltime, tsorted, R); one per stream


def get_delta_t(src_event, neg_dst_event, t_event, neighbors, times, offsets, default=1e+11):
    """
    Vectorized delta-t for (src, neg_dst) evaluated at time t_event (replaces a
    per-element Python loop with ~4*B device syncs per window).

    Semantics: for each position, delta = t - max time of any *direct*
    src<->dst interaction strictly before t, else `default`; then log1p.
    Bit-exact vs the old per-element loop for every input (adversarial
    time-interleaved harness on real repeated pairs).

    Args:
        src_event     : LongTensor [B]  source nodes of events
        neg_dst_event : LongTensor [B]  negative destinations
        t_event       : LongTensor [B]  event times
        neighbors     : [2E] neighbor list from build_neighbor_index_torch
        times         : [2E] interaction times aligned with neighbors
        offsets       : LongTensor [N+1] index offsets for neighbors

    Returns:
        delta_t : FloatTensor [B]  log1p(time since last src–neg_dst interaction)
    """
    device = neighbors.device
    cache_key = id(neighbors)
    if cache_key not in _PAIR_CACHE:
        _PAIR_CACHE[cache_key] = _build_pair_table(neighbors, times, offsets)
    lex, ltime, tsorted, R = _PAIR_CACHE[cache_key]
    N = offsets.numel() - 1

    u = src_event.long()
    v = neg_dst_event.long()
    q = torch.minimum(u, v) * N + torch.maximum(u, v)

    # rank of the query time in the global time order; entries with
    # time_rank < rq are exactly those with time < t
    tq = t_event.reshape(-1).to(ltime.dtype)
    rq = torch.searchsorted(tsorted, tq, side='left')

    # lexicographic (pair, time) search: j = first entry with key == pair
    # AND time_rank >= rq (converting to "no interaction before t" when none)
    lo = torch.searchsorted(lex, q * R, side='left')            # segment start
    hi = torch.searchsorted(lex, (q + 1) * R, side='left')      # segment end
    j = torch.searchsorted(lex, q * R + rq, side='left')        # global cut

    # valid: at least one interaction of this pair lies strictly before t
    valid = (j > lo) & (j <= hi)
    last_t_raw = ltime[(j - 1).clamp_min(0)]

    B = src_event.size(0)
    delta_t = torch.where(
        valid,
        tq - last_t_raw,          # j-1 is a same-pair entry with time < t ⇒ >= 0
        torch.full((B,), default, dtype=torch.float32, device=device),
    )
    return torch.log1p(delta_t)