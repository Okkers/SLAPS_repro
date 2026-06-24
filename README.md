# SLAPS_repro

This is the code repository associated with our reproduction article [*Reproduction of "SLAPS: Self-Supervision Improves Structure Learning for Graph Neural Networks" (Fatemi et al.)*](https://hackmd.io/@l2ShhaYtT8K79WzeUtmyUQ/rJhZkOYffe). The repo contains the training code, dataset loaders, experiment scripts, and utilities for reproducing all figures, models and numbers in the article.


<img width="857" height="317" alt="Screenshot 2026-06-16 023947(1)" src="https://github.com/user-attachments/assets/660334b1-2184-4027-acc2-5bb22aec1a97" />

(*Figure from the original paper, of the SLAPS architecture*)

## Setup

Create a virtual environment and install the Python dependencies:

```bash
pip install -r requirements.txt
```

Requirements are listed in `requirements.txt` and include:

- `torch`
- `torch-geometric`
- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `pandas`
- `ogb`

### Data requirements
The code requires various datasets from various sources. All links to the datasets we used can also be found in the [original paper](https://proceedings.neurips.cc/paper/2021/hash/bf499a12e998d178afd964adf64a60cb-Abstract.html). Before downloading any datasets, run ```mkdir data``` from root. 

-  `cora`, `citeseer`, `pubmed`, `cora390`, `citeseer370`expect the standard Planetoid-style files in `data/` such as `ind.cora.x`, `ind.cora.y`, and the related `tx/ty/allx/ally/graph/test.index` files.

- `ogbn-arxiv` is loaded through `ogb` and will download its data into `data/` when needed.
- `mnist` is downloaded automatically through `torchvision`.
- `wine`, `cancer`, `digits`, and `20news` are loaded from `scikit-learn`.

GPU acceleration is possible (and recommended for the larger graph benchmarks), with CUDA-compatible graphics cards.

## Repository Layout

- `data_provider/`: dataset loading and split creation for citation graphs, OGBN-Arxiv, MNIST, and tabular datasets.

- `exp/`: training loop, evaluation, optimizer setup, and early stopping.

- `layers/`: model building blocks.

- `models/`: SLAPS model variants:
  - `SLAPS_FP`
  - `SLAPS_MLP`
  - `SLAPS_MLPD`

- `utils/`: helper functions, metrics, graph utilities, and figure-generation scripts.

- `scripts/`: shell scripts for the different datasets and model variants.

## Running `main.py`

`main.py` requires the following arguments:

- `--is_training`
- `--model_id`
- `--model`
- `--train_epochs`
- `--dataset`

An example call (which can also be found in different flavors in `scripts/`) looks like this:

```bash
python -u main.py \
  --is_training 1 \
  --train_epochs 2000 \
  --model_id CORA_TEST \
  --dataset cora \
  --model SLAPS_MLP \
  --input_dim 1433 \
  --hidden_dim 32 \
  --output_dim 7 \
  --r 10 \
  --eta 5 \
  --noise_type zero \
  --generator MLP \
  --gen_input_dim 1433 \
  --gen_layers_size 2 \
  --gen_k 20 \
  --use_gpu True \
  --lr_c 0.01 \
  --lr_DAE 0.001 \
  --dropout_c 0.25 \
  --dropout_DAE 0.5 \
  --lambda_val 10 \
  --itr 10 \
  --patience 500 \
  --is_discrete \
  --weight_decay_c 0.0005 \
  --hidden_dim_dae 512
```

## Figures in `utils/`

### Reproduction vs. original paper

`utils/repro_v_orig.py` builds the comparison heatmaps between the original paper results and the reproduction results.

Run it from the repo root:

```bash
python utils/repro_v_orig.py
```

This writes:

- `repro_vs_original_diff.png`

### Homophily odds plot

`utils/plot_homophily_odds.py` compares learned adjacency homophily odds for `cora` and `citeseer` checkpoints (which does require trained checkpoints!)

Example:

```bash
python utils/plot_homophily_odds.py \
  --checkpoint-cora ./checkpoints/cora_run \
  --checkpoint-citeseer ./checkpoints/citeseer_run \
  --output homophily_odds.png
```

This writes the figure you specify with `--output`.

## Branches

Two branches are kept separate, `feature/ada-edge-self-training` and `feature/fig_3,4,5_reproduction`, with the first one containing cluster scripts for the datasets too large for consumer hardware (obgn and citeseer), and the second for reproducing experiments tied to figures 3, 4 and 5 in the original paper. They differ in a few training and data-processing steps so the code can match different experiment configurations. If you are reproducing a specific result, make sure you are on the branch whose implementation matches that configuration!
