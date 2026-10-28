
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("Changed working directory to:", os.getcwd())

import random
import argparse
import json
import yaml
import torch

import numpy as np
from utils.DataLoader import get_link_prediction_data

import torch.optim as optim
from ssm_utlis import positional_encoding_mixer,add_edge_features,set_seed,combine_sort,aggregate_metrics


from ssm_memory import MemoryModel

from ssm_dataset import TemporalEdgeDataset
from ssm_learn import main


def load_config(config_path):
    """Load JSON or YAML config file."""
    if config_path.endswith(".json"):
        with open(config_path, "r") as f:
            return json.load(f)
    elif config_path.endswith((".yml", ".yaml")):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Config file must be .json, .yml or .yaml")

def enforce_types(args):
    """Ensure argparse + config values have the correct Python types."""
    args.lr = float(args.lr)
    args.wd = float(args.wd)
    args.val_ratio = float(args.val_ratio)
    args.test_ratio = float(args.test_ratio)

    args.time_dim = int(args.time_dim)
    args.hidden_dim = int(args.hidden_dim)
    args.embd_dim = int(args.embd_dim)
    args.batches = int(args.batches)
    args.context = int(args.context)
    args.epochs = int(args.epochs)
    args.runs = int(args.runs)

    args.reg = float(args.reg)
    args.delta = float(args.delta)
    args.u_batch = bool(args.u_batch)
    return args

def run_training(args):
    # -----------------------
    # Dataset Preparation
    # -----------------------
    
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_prediction_data(dataset_name=args.dataset, val_ratio=args.val_ratio, test_ratio=args.test_ratio, edge_pad=False)
    
    

    # Merge New Node with mask
    combined_val = combine_sort(val_data,new_node_val_data)
    combined_test = combine_sort(test_data,new_node_test_data)


    # Time normalization + encoding
    # edge_raw_features = #edge_raw_features[]   # remove first invalid entry
    scale = 3600
    timestamps = train_data.node_interact_times / scale
    combined_test.node_interact_times = combined_test.node_interact_times/scale
    combined_val.node_interact_times = combined_val.node_interact_times/scale
    full_data.node_interact_times = full_data.node_interact_times / scale
 
    # Merge train + val
    c_time = np.hstack((timestamps, combined_val.node_interact_times))
    c_src = np.hstack((train_data.src_node_ids, combined_val.src_node_ids))
    c_dst = np.hstack((train_data.dst_node_ids, combined_val.dst_node_ids))


    # Special handling for certain datasets
    if args.dataset in ['enron', 'uci','lastfm']:
        edge_raw_features += 1
        edge_raw_features = edge_raw_features.mean(axis=-1)[:, None]
    elif args.dataset in ['mooc']:
        edge_raw_features = edge_raw_features[:,:4]
    
    elif args.dataset in ['SocialEvo']:
        edge_raw_features = edge_raw_features[:,:2]
    

    if args.dataset in {'li_small', 'li_medium', 'li_large'}:
        # IBM-AML datasets: edge features are PRE-BAKED offline by
        # preprocess/bake_features.py into ml_<name>.npy as [N+1, 172] float32
        # (cols = raw | cos-time-enc(f2) | f1, row 0 = dummy zeros).
        # Skip add_edge_features entirely (it would re-materialize ~10 GB float64).
        assert edge_raw_features.shape[1] == 172
        tmx = np.asarray(np.load(   # per-edge delta-t (f2), [N+1] float32, edge_ids 1..N
            './processed_data/{}/ml_{}_f2.npy'.format(args.dataset, args.dataset),
            mmap_mode='r'
        ))
    else:
        # Add edge features
        edge_raw_features, tmx = add_edge_features(
            full_data.src_node_ids,
            full_data.dst_node_ids,
            full_data.node_interact_times,
            edge_raw_features[1:],
            positional_encoding_mixer,
            args.time_dim
        )

        #add_dud
        edge_feat_dim = edge_raw_features.shape[1]
        zero_padd = np.zeros((1,edge_feat_dim))
        edge_raw_features = np.vstack((zero_padd,edge_raw_features))
        fin_tmx = np.zeros((len(tmx)+1,))
        fin_tmx[1:] = tmx
        tmx = fin_tmx

    edge_feat_dim = edge_raw_features.shape[1]


    total_nodes = node_raw_features.shape[0]

    if args.dataset in ['lastfm','reddit','wikipedia']:
        biparted=True
    else :
        biparted = False

    # Dataset wrapper
    TGD = TemporalEdgeDataset(
        train_data.src_node_ids,
        train_data.dst_node_ids,
        train_data.node_interact_times,
        edge_raw_features[train_data.edge_ids],
        biparted=biparted,
        T=args.batches,
        num_node=total_nodes,
        LastInteraction=tmx[train_data.edge_ids],
        context_window=args.context,
        mode='train',
        device=args.device,
        unique_batch=args.u_batch,
        negative= args.negative 
    )
    # torch.autograd.set_detect_anomaly(True)
    inductive_key = len(train_data.dst_node_ids)
    # Dataset wrapper
    TGD_val = TemporalEdgeDataset(
        combined_val.src_node_ids,
        combined_val.dst_node_ids,
        combined_val.node_interact_times,
        edge_raw_features[combined_val.edge_ids],
        biparted=biparted,
        T= args.batches,
        num_node=total_nodes,
        LastInteraction=tmx[combined_val.edge_ids],
        context_window=args.context,
        mode='eval',
        inductive_cut=inductive_key,
        device=args.device,
        unique_batch=args.u_batch,
        negative= args.negative,
        c_dst=c_dst,
        c_src=c_src,
        c_time=c_time,
        inductive_mask=combined_val.inductive_mask 
    )

    c_dst = np.hstack((c_dst,combined_test.dst_node_ids))
    c_src = np.hstack((c_src,combined_test.src_node_ids))
    c_time = np.hstack((c_time,combined_test.node_interact_times))
    
    u,v = list(test_data.src_node_ids),list(test_data.dst_node_ids)
    test_edge = set((ub,vb) for ub,vb in zip(u,v))
    TGD_test = TemporalEdgeDataset(
    combined_test.src_node_ids,
    combined_test.dst_node_ids,
    combined_test.node_interact_times,
    edge_raw_features[combined_test.edge_ids],
    biparted=biparted,
    T=args.batches,
    num_node=total_nodes,
    LastInteraction=tmx[combined_test.edge_ids],
    context_window=args.context,
    mode='eval',
    inductive_cut=inductive_key,
    device=args.device,
    unique_batch=args.u_batch,
    negative= args.negative,
    c_dst=c_dst,
    c_src=c_src,
    c_time=c_time,
    inductive_mask=combined_test.inductive_mask,
    test_edge_set=test_edge
    )

    print(f'Eval NSS: {args.negative}')
    print("Arguments used:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print(f'Training/ Testing model for {args.runs} runs')

    T_val_results = []
    I_val_results = []
    for run in range(args.runs):
        # set_seed(run)

        # -----------------------
        # Model
        # -----------------------
        # seed = 42
        # torch.manual_seed(seed)
        # np.random.seed(seed)
        # random.seed(seed)
    
        model = MemoryModel(
            num_nodes=total_nodes,
            input_dim=edge_feat_dim,
            hidden_dim=args.hidden_dim,
            time_dim=args.time_dim,
            reg=args.reg,
            delta=args.delta,
            device=args.device,
            embd_dims=args.embd_dim,
            update_type='mamba',
            use_old_message=False
        ).to(args.device)

        torch.seed()         # resets torch RNG
        np.random.seed(None) # resets numpy RNG
        random.seed()        # resets Python RNG

        #  Count only trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Trainable parameters:", trainable_params)
        # Optimizer + training
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        results = main(
            model=model,
            val_dataset=TGD_val,
            test_dataset=TGD_test,
            train_dataset=TGD,
            optimizer=optimizer,
            name=args.name,
            num_epochs=args.epochs,
            device=args.device
        )
        T_val_results.append(results['transductive'])
        I_val_results.append(results['inductive'])


    print("\n==== Transductive Test Metrics Across Runs ====")
    t_summary = aggregate_metrics(T_val_results, 'transductive')
    for neg_type in ['rnd', 'hst', 'ind']:
        print(f"{neg_type.upper()} - AUC: {t_summary[neg_type]['AUC_mean']:.4f} ± {t_summary[neg_type]['AUC_std']:.4f} | "
            f"AP: {t_summary[neg_type]['AP_mean']:.4f} ± {t_summary[neg_type]['AP_std']:.4f}")
        
    print("\n==== Inductive Test Metrics Across Runs ====")
    i_summary = aggregate_metrics(I_val_results, 'inductive')
    for neg_type in ['rnd', 'hst', 'ind']:
        print(f"{neg_type.upper()} - AUC: {i_summary[neg_type]['AUC_mean']:.4f} ± {i_summary[neg_type]['AUC_std']:.4f} | "
            f"AP: {i_summary[neg_type]['AP_mean']:.4f} ± {i_summary[neg_type]['AP_std']:.4f}")

    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal Link Prediction Training")

    # Config file option
    parser.add_argument("--config", type=str, help="Path to YAML/JSON config file", default=None)

    # Dataset / files
    parser.add_argument("--dataset", type=str, default="lastfm", help="Dataset name")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)

    # Model params
    parser.add_argument("--time_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--embd_dim", type=int, default=16)

    parser.add_argument("--context", type=int, default=10)
    parser.add_argument("--reg", type=float, default=1e-2)
    parser.add_argument("--delta", type=float, default=1e-3)

    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)

    # Training
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--u_batch", type=bool, default=False)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--negative", type=str, default='RNS')

    # Misc
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--name", type=str, default="Experiment_")

    args = parser.parse_args()

    # If config file is provided, load it and override defaults
    if args.config is not None:
        config_dict = load_config(args.config)
        for k, v in config_dict.items():
            setattr(args, k, v)
    args = enforce_types(args)
    run_training(args)
    