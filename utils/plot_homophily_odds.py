import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_provider.data_loader import load_citation_data
from models import SLAPS_FP, SLAPS_MLP, SLAPS_MLPD

WEIGHT_BINS = [
    (0.0, 0.0, "[0.0, 0.0]"),
    (0.0, 0.01, "(0.0, 0.01)"),
    (0.01, 0.03, "[0.01, 0.03)"),
    (0.03, 0.05, "[0.03, 0.05)"),
    (0.05, 0.08, "[0.05, 0.08)"),
    (0.08, 0.10, "[0.08, 0.1)"),
    (0.10, float("inf"), "[0.1, inf)"),
]

def build_args(dataset, checkpoint_dir, use_gpu):
    class Args:
        pass

    args = Args()
    args.is_training = 0
    args.model_id = "PLOT"
    args.model = "SLAPS_MLP"
    args.train_epochs = 1
    args.dataset = dataset
    args.mnist_training_size = 1000
    args.checkpoints = checkpoint_dir
    args.itr = 1
    args.patience = 10
    args.input_dim = 1433 if dataset == "cora" else 3703
    args.hidden_dim = 32
    args.output_dim = 7 if dataset == "cora" else 6
    args.is_discrete = True
    args.r = 10
    args.eta = 5
    args.noise_type = "zero" if dataset == "cora" else "not_zero"
    args.generator = "MLP"
    args.gen_input_dim = args.input_dim
    args.gen_layers_size = 2
    args.gen_k = 20 if dataset == "cora" else 30
    args.use_gpu = use_gpu and torch.cuda.is_available()
    args.lr_c = 0.01
    args.dropout_c = 0.25
    args.dropout_DAE = 0.5
    args.lr_DAE = 0.001
    args.lambda_val = 10
    args.weight_decay_c = 0.0005
    args.weight_decay_dae = 0.0
    args.hidden_dim_dae = 512 if dataset == "cora" else 1024
    args.gpu = 0
    args.use_multi_gpu = False
    args.devices = ""
    return args


def infer_model_class(state_dict):
    keys = set(state_dict.keys())
    if "generator.A" in keys:
        return SLAPS_FP, "SLAPS_FP"

    if any(key.startswith("generator.mlp.") and key.endswith(".weight") for key in keys):
        return SLAPS_MLP, "SLAPS_MLP"

    if any(key.startswith("generator.mlp.") for key in keys):
        return SLAPS_MLPD, "SLAPS_MLPD"

def load_model(args, checkpoint_path, device):
    features, labels, _, _, test_mask, _, _ = load_citation_data(args.dataset)
    args.features = features

    state = torch.load(checkpoint_path, map_location=device)
    model_cls, model_name = infer_model_class(state)
    args.model = model_name
    model = model_cls.Model(args).float().to(device)
    model.load_state_dict(state)
    model.eval()
    return model, features.to(device), labels.to(device), test_mask.to(device)


def learned_adjacency(model, features):
    with torch.no_grad():
        raw_adj = model.generator(features)
        adj = model.adjacency_processor.apply_adj_processor(raw_adj)
    return adj.detach().cpu()


def compute_odds_by_bin(adj, labels, test_mask):
    test_idx = torch.where(test_mask)[0].cpu().numpy()
    sub_adj = adj[np.ix_(test_idx, test_idx)]
    sub_labels = labels[test_mask].cpu().numpy()

    tri_i, tri_j = np.triu_indices(len(test_idx), k=1)
    weights = sub_adj[tri_i, tri_j]
    same = sub_labels[tri_i] == sub_labels[tri_j]

    odds = []
    counts = []
    for low, high, _ in WEIGHT_BINS:
        if np.isinf(high):
            mask = weights >= low
        elif low == 0.0 and high == 0.0:
            mask = weights == 0.0
        else:
            mask = (weights > low) & (weights < high)

        same_count = int(same[mask].sum())
        diff_count = int(mask.sum() - same_count)
        odds.append(np.nan if diff_count == 0 else same_count / diff_count)
        counts.append(int(mask.sum()))

    return odds, counts


def average_curves(curves):
    stacked = np.array(curves, dtype=float)
    with np.errstate(invalid="ignore"):
        means = np.nanmean(stacked, axis=0)
    counts = np.sum(np.isfinite(stacked), axis=0).astype(int)
    means[counts == 0] = np.nan
    return means, counts


def resolve_checkpoints(root):
    root_path = Path(root)
    if root_path.is_file():
        return [root_path]

    checkpoints = list(root_path.rglob("checkpoint*.pth"))

    def checkpoint_sort_key(path):
        stem = path.stem
        if stem == "checkpoint":
            return (0, -1, str(path))

        if stem.startswith("checkpoint(") and stem.endswith(")"):
            inner = stem[len("checkpoint("):-1]
            if inner.isdigit():
                return (0, int(inner), str(path))

        return (1, stem, str(path))

    unique = sorted({path.resolve() for path in checkpoints}, key=checkpoint_sort_key)
    return unique

def plot_results(results, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    x = np.arange(len(WEIGHT_BINS))
    labels = [label for _, _, label in WEIGHT_BINS]

    style_map = {
        "cora": {"color": "red", "linestyle": "-", "marker": "o", "label": "Cora"},
        "citeseer": {"color": "blue", "linestyle": ":", "marker": "s", "label": "Citeseer"},
    }

    for dataset, stats in results.items():
        odds = stats["odds"]
        style = style_map.get(dataset, {})
        ax.plot(
            x,
            odds,
            linewidth=2,
            markersize=6,
            **style,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("Edge weight interval")
    ax.set_ylabel("Odds")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-cora", required=True)
    parser.add_argument("--checkpoint-citeseer", required=True)
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--output", default="homophily_odds.png")
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu")

    results = {}
    for dataset, ckpt in [("cora", args.checkpoint_cora), ("citeseer", args.checkpoint_citeseer)]:
        model_args = build_args(dataset, args.checkpoint_dir, args.use_gpu)
        checkpoint_paths = resolve_checkpoints(ckpt)
        print(f"{dataset}: found {len(checkpoint_paths)} checkpoint(s)")

        run_odds = []
        run_counts = []

        for checkpoint_path in checkpoint_paths:
            model, features, labels, test_mask = load_model(model_args, checkpoint_path, device)
            adj = learned_adjacency(model, features)
            odds, counts = compute_odds_by_bin(adj, labels, test_mask)
            run_odds.append(odds)
            run_counts.append(counts)

        avg_odds, _ = average_curves(run_odds)
        results[dataset] = {"odds": avg_odds}

    plot_results(results, args.output)


if __name__ == "__main__":
    main()
