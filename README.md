
# Deep Learning Experiment Template

A modular, reproducible deep learning template designed for seamless transition between local development and **Slurm** clusters.

**Tech Stack:**

* **[uv](https://github.com/astral-sh/uv):** Blazing fast Python package management and virtual environment handling.
* **[Hydra](https://hydra.cc/):** Compositional configuration management.
* **[Submitit](https://github.com/facebookincubator/submitit):** Slurm job submission directly from Python.
* **[Weights & Biases](https://wandb.ai/):** Experiment tracking and visualization.

---

## Repository Structure

```text
├── configs/                 # Hydra configuration files
│   ├── config.yaml          # Main config (defaults)
│   ├── model/               # Model architecture hyperparameters
│   └── hydra/launcher/      # Slurm submission settings
├── data/                    # Data storage (git-ignored)
├── logs/                    # Local logs and Slurm output files
├── scripts/                 # Executable entry points
│   └── train.py             # Main training script
├── src/                     # Source code (installed as editable package)
│   └── my_model/            # Your actual python package
│       ├── models/          # PyTorch modules
│       └── utils/           # Utilities (W&B logging, etc.)
├── pyproject.toml           # Dependency definitions
└── uv.lock                  # Exact dependency versions for reproducibility

```

---

## Quick Start

### 1. Prerequisites

You need **[uv](https://github.com/astral-sh/uv)** installed.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Installation

Initialize the environment and install dependencies (including the local `src` package).

```bash
uv sync
```

### 3. Weights & Biases Setup

Ensure you are logged in to tracking experiments.

```bash
uv run wandb login
```

---

## Running Experiments

This repository uses **Hydra** to handle configurations. You can override any parameter from the command line.

### Option A: Local Run

Good for debugging and small-scale testing.

```bash
# Run with defaults

uv run python scripts/train.py

# Override hyperparameters
uv run python scripts/train.py model.num_classes=100 epochs=5
```

### Option B: Slurm Cluster Run

Submit jobs to the cluster directly from your workstation (or head node) without writing `.sbatch` files. The `submitit` plugin handles the job submission.

**Note:** The `--multirun` flag is required to trigger the launcher plugin, even for single jobs.

```bash
# Submit a single job to Slurm
uv run python scripts/train.py --multirun

# Run a Hyperparameter Sweep (runs 2 jobs with different LRs)
uv run python scripts/train.py --multirun lr=1e-3,1e-4
```

---

## Configuration & Slurm Settings

### modifying Slurm Parameters

To change partition, time limits, or GPU requests, edit `configs/hydra/launcher/submitit_slurm.yaml`.

**Key Fields:**

* `partition`: The cluster partition to use (default: `gpu`).

* `timeout_min`: Max runtime in minutes.

* `gpus_per_node`: Number of GPUs requested.

* `setup`: Bash commands to run before Python starts (e.g., `module load cuda`).

**Example Override via CLI:**

```
uv run python scripts/train.py --multirun \
    hydra.launcher.partition=high_priority \
    hydra.launcher.timeout_min=60
```

---

## Development Workflow

1. **Add Dependencies:**

```bash
uv add matplotlib
```

2. **Format Code:**

```bash
uv run ruff check .
```

3. **Run Tests:**

```bash
uv run pytest
```

## Troubleshooting

* **`Interpolation key 'hydra.job.name' not found`:**

  * This happens if you removed the `oc.select` safety wrapper in the launcher config. Ensure your config uses `${oc.select:hydra.job.name,local_run}` to handle both local and cluster contexts.

* **W&B Offline Mode:**

  * If compute nodes have no internet, edit `configs/config.yaml` to set `wandb.mode: "offline"`. You can sync runs later using `wandb sync wandb/run-...`.
