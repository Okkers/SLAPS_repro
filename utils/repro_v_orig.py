import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Cell:
    mean: Optional[float]
    std: Optional[float] = None
    status: Optional[str] = None

    def is_missing(self) -> bool:
        return self.mean is None


def ok(mean, std=None):
    return Cell(mean=float(mean), std=None if std is None else float(std))

def missing(status):
    return Cell(mean=None, std=None, status=status)

GRAPH_MODELS = [
    "SLAPS (FP)",
    "SLAPS (MLP)",
    "SLAPS (MLP-D)",
    "SLAPS (MLP) + AdaEdge",
    "SLAPS (MLP) + self-training",
]

GRAPH_DATASETS = ["Cora", "Citeseer", "Cora390", "Citeseer370", "Pubmed", "ogbn-arxiv"]

TABULAR_MODELS = ["SLAPS (FP)", "SLAPS (MLP)", "SLAPS (MLP-D)"]
TABULAR_DATASETS = ["Wine", "Cancer", "Digits", "20news"]

MNIST_MODELS = ["SLAPS"]
MNIST_DATASETS = ["MNIST1000", "MNIST2000", "MNIST3000"]

ORIGINAL_RESULTS = {
    "Graph benchmarks": {
        "models": GRAPH_MODELS,
        "datasets": GRAPH_DATASETS,
        "values": {
            "SLAPS (FP)": [ok(72.4, 0.4), ok(70.7, 0.4), ok(76.6, 0.4), ok(73.1, 0.6), missing("OOM"), missing("OOM")],
            "SLAPS (MLP)": [ok(72.8, 0.8), ok(70.5, 1.1), ok(75.3, 1.0), ok(73.0, 0.9), ok(74.4, 0.6), ok(56.6, 0.1)],
            "SLAPS (MLP-D)": [ok(73.4, 0.3), ok(72.6, 0.6), ok(75.1, 0.5), ok(73.9, 0.4), ok(73.1, 0.7), ok(52.9, 0.1)],
            "SLAPS (MLP) + AdaEdge": [ok(72.8, 0.7), ok(70.6, 1.5), ok(75.2, 0.6), ok(72.6, 1.4), missing("OOT"), missing("OOT")],
            "SLAPS (MLP) + self-training": [ok(74.2, 0.5), ok(73.1, 1.0), ok(75.5, 0.7), ok(73.3, 0.6), ok(74.3, 1.4), missing("NA")],
        },
    },
    "Tabular benchmarks": {
        "models": TABULAR_MODELS,
        "datasets": TABULAR_DATASETS,
        "values": {
            "SLAPS (FP)": [ok(96.6, 0.4), ok(94.6, 0.3), ok(94.4, 0.7), ok(44.4, 0.8)],
            "SLAPS (MLP)": [ok(96.3, 1.0), ok(96.0, 0.8), ok(92.5, 0.7), ok(50.4, 0.7)],
            "SLAPS (MLP-D)": [ok(96.5, 0.8), ok(96.6, 0.2), ok(94.2, 0.1), ok(49.8, 0.9)],
        },
    },
    "MNIST benchmarks": {
        "models": MNIST_MODELS,
        "datasets": MNIST_DATASETS,
        "values": {
            "SLAPS": [ok(94.66, 0.2), ok(95.35, 0.1), ok(95.54, 0.0)],
        },
    },
}

REPRO_RESULTS = {
    "Graph benchmarks": {
        "models": GRAPH_MODELS,
        "datasets": GRAPH_DATASETS,
        "values": {
            "SLAPS (FP)": [ok(69.96, 0.8), ok(69.87, 0.52), ok(74.37, 0.6), ok(73.00, 0.37), missing("OOM"), missing("OOM")],
            "SLAPS (MLP)": [ok(69.60, 0.6), ok(69.27, 1.46), ok(73.56, 0.94), ok(72.94, 0.85), ok(68.4, 1.2), missing("OOM")],
            "SLAPS (MLP-D)": [ok(71.02, 0.35), ok(70.94, 1.18), ok(75.13, 1.02), ok(71.66, 0.85), ok(69.8, 0.8), missing("OOM")],
            "SLAPS (MLP) + AdaEdge": [ok(69.6, 0.9), ok(69.1, 1.5), ok(73.7, 0.7), ok(72.7, 0.8), missing("OOT"), missing("OOT")],
            "SLAPS (MLP) + self-training": [ok(71.3, 0.6), ok(70.2, 0.6), ok(73.8, 1.2), ok(73.3, 0.6), ok(65.6, 1.4), missing("NA")],
        },
    },
    "Tabular benchmarks": {
        "models": TABULAR_MODELS,
        "datasets": TABULAR_DATASETS,
        "values": {
            "SLAPS (FP)": [ok(96.7, 0.6), ok(96.0, 0.6), ok(91.8, 0.7), ok(42.5, 0.7)],
            "SLAPS (MLP)": [ok(94.0, 0.8), ok(93.0, 2.3), ok(91.2, 1.0), ok(43.4, 0.8)],
            "SLAPS (MLP-D)": [ok(96.3, 0.5), ok(90.1, 0.8), ok(91.5, 0.7), ok(44.0, 0.4)],
        },
    },
    "MNIST benchmarks": {
        "models": MNIST_MODELS,
        "datasets": MNIST_DATASETS,
        "values": {
            "SLAPS": [ok(92.0, 1.2), ok(92.2, 0.3), ok(94.3, 0.5)],
        },
    },
}

def diff_matrix(models, datasets, original_values, repro_values):
    matrix = np.full((len(models), len(datasets)), np.nan, dtype=float)
    annotations = [["" for _ in datasets] for _ in models]
    missing_labels = [["" for _ in datasets] for _ in models]

    for i, model in enumerate(models):
        original_row = original_values.get(model, [])
        repro_row = repro_values.get(model, [])
        for j in range(len(datasets)):
            original_cell = original_row[j]
            repro_cell = repro_row[j]

            if original_cell.is_missing() or repro_cell.is_missing():
                if original_cell.is_missing() and repro_cell.is_missing():
                    if original_cell.status == repro_cell.status:
                        label = original_cell.status or "NA"
                    else:
                        label = f"{original_cell.status or 'NA'} / {repro_cell.status or 'NA'}"
                elif original_cell.is_missing():
                    label = f"orig: {original_cell.status or 'NA'}"
                else:
                    label = f"repro: {repro_cell.status or 'NA'}"
                missing_labels[i][j] = label
                continue

            delta = repro_cell.mean - original_cell.mean
            matrix[i, j] = delta
            annotations[i][j] = f"{delta:+.2f}".rstrip("0").rstrip(".")

    return matrix, annotations, missing_labels

def text_color_for_value(value, cmap, vmin, vmax):
    rgba = cmap((value - vmin) / (vmax - vmin))
    r, g, b = rgba[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.5 else "black"

def draw_panel(ax, title, models, datasets, original_values, repro_values, color_limit, show_ylabel=False):
    matrix, annotations, missing_labels = diff_matrix(models, datasets, original_values, repro_values)
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="#d9d9d9")

    if color_limit == 0:
        color_limit = 1.0

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-color_limit, vmax=color_limit)
    ax.set_title(title, pad=10, fontweight="bold", fontsize=18)
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(datasets, rotation=30, fontsize=14, ha="right", fontname="monospace")
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models if show_ylabel else [""] * len(models), fontsize=14, fontname="monospace")
    ax.tick_params(axis="both", which="both", length=0)

    ax.set_xticks(np.arange(-0.5, len(datasets), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(models)):
        for j in range(len(datasets)):
            if np.isfinite(matrix[i, j]):
                color = text_color_for_value(matrix[i, j], cmap, -color_limit, color_limit)
                ax.text(
                    j,
                    i,
                    annotations[i][j],
                    ha="center",
                    va="center",
                    fontsize=14,
                    color=color,
                    fontname="monospace",
                )
            else:
                label = missing_labels[i][j]
                if label:
                    ax.text(
                        j,
                        i,
                        label,
                        ha="center",
                        va="center",
                        fontsize=14,
                        color="#333333",
                        fontname="monospace",
                    )

    return im

def build_figure(output_path):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(18, 18), constrained_layout=True)
    fig.suptitle("Reproduction minus original paper", fontsize=24, fontweight="bold", fontname="monospace")

    families = [
        ("Graph benchmarks", 0),
        ("Tabular benchmarks", 1),
        ("MNIST benchmarks", 2),
    ]

    all_deltas = []
    for family_name, _ in families:
        original = ORIGINAL_RESULTS[family_name]
        repro = REPRO_RESULTS[family_name]
        matrix, _, _ = diff_matrix(
            original["models"],
            original["datasets"],
            original["values"],
            repro["values"],
        )
        all_deltas.append(matrix[np.isfinite(matrix)])

    finite_all = np.concatenate([arr for arr in all_deltas if arr.size]) if any(arr.size for arr in all_deltas) else np.array([1.0])
    color_limit = float(np.max(np.abs(finite_all)))
    if color_limit == 0:
        color_limit = 1.0

    images = []
    for family_name, row in families:
        original = ORIGINAL_RESULTS[family_name]
        repro = REPRO_RESULTS[family_name]
        images.append(
            draw_panel(
                axes[row],
                f"{family_name}: difference",
                original["models"],
                original["datasets"],
                original["values"],
                repro["values"],
                color_limit,
                show_ylabel=True,
            )
        )

    cbar = fig.colorbar(images[0], ax=axes, shrink=0.9, pad=0.02, label="Reproduction - original (%)")
    cbar.ax.tick_params(length=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure to {output_path}")

if __name__ == "__main__":
    build_figure("repro_vs_original_diff.png")
