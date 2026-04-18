# Anomalous-Machine-Sound-Detection

## Overview

This project aims to detect abnormal machine sounds using deep learning techniques on the MIMII dataset. The task is framed as a binary classification problem: `normal` vs `abnormal` audio.

The goal is not only to build a working model, but to create a **reproducible experimental pipeline** that allows comparison of:
- different preprocessing methods
- different model architectures
- different hyperparameters

For a deeper explanation of the model design and pipeline decisions, see:
- [Baseline design documentation](docs/baseline.md)

---

## Setup

Before running the project, please follow the setup instructions:

- 🔧 [Setup guide](docs/setup.md)

This includes:
- Python environment setup
- PyTorch and TorchAudio installation
- dataset setup instructions

---

## Quick Start (Baseline Workflow)
This project provides a reproducible baseline pipeline for abnormal machine sound detection.

The commands below run the full baseline workflow end-to-end.


For a detailed explanation of each step, see:
- [Baseline usage guide](docs/baseline_usage.md)

The baseline CNN workflow is the primary supported path in this repository.

### Step 1 — Generate a split manifest
```bash
python scripts/make_splits.py --machine-type fan --seed 42
```
For details on how splits are constructed and why group-based splitting is used, see:
- [Data and split design](docs/data.md)
### Step 2 — Train the baseline model
```bash
python scripts/train.py \
  --manifest-path data/splits/fan_split_seed42.csv \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 1e-3 \
  --run-name fan_baseline
```
For full training options and advanced usage (e.g. resume training), see:
- [Baseline training guide](docs/baseline_usage.md)
  
### Step 3 — Evaluate the trained model
```bash
python scripts/evaluate.py \
  --manifest-path data/splits/fan_split_seed42.csv \
  --checkpoint-path artifacts/checkpoints/fan_baseline/best.pt \
  --run-name fan_baseline
```
For details on evaluation metrics and implementation, see:
- [Evaluation and metrics](docs/baseline_usage.md)
### Step 4 — Generate plots
```bash 
python scripts/plot_results.py \
  --history-path artifacts/metrics/fan_baseline/history.json \
  --metrics-path artifacts/metrics/fan_baseline/test_metrics.json \
  --run-name fan_baseline
```
For details on available plots and customization, see:
- [Plotting and visualization](docs/baseline_usage.md)


### Expected outputs
The following artifacts will be generated:
```text
artifacts/checkpoints/fan_baseline/best.pt
artifacts/metrics/fan_baseline/test_metrics.json
artifacts/curves/fan_baseline/loss_curve.png
artifacts/metrics/fan_baseline/history.json
```

## Where to go next

- Want to understand the model? → [Baseline design](docs/baseline.md)
- Want to modify training? → [Baseline usage](docs/baseline_usage.md)
- Want to reproduce results exactly? → [Reproducibility guide](docs/reproducibility.md)

---

## Features

- PyTorch-based training pipeline
- Reproducible train / validation / test split manifests
- Raw WAV dataset loading from saved split manifests
- Deterministic on-the-fly preprocessing for the baseline
- Baseline CNN training with checkpointing
- Test-set evaluation with saved metrics
- Plotting utilities for training curves and confusion matrix
- Ability to:
  - train from scratch
  - resume from a saved checkpoint
  - evaluate the saved best checkpoint
  - reproduce reported baseline results

---

## Project Structure

```text
project_root/
├── .gitignore
├── README.md
├── requirements.txt

├── data/
│   ├── raw/                     # original downloaded audio files
│   ├── processed/               # optional cached/precomputed features
│   └── splits/                  # saved train/val/test split manifests

├── artifacts/
│   ├── checkpoints/             # saved model checkpoints by run
│   ├── curves/                  # saved plots by run
│   └── metrics/                 # saved training/evaluation metrics by run

├── docs/
│   ├── autoencoder_usage.md     # autoencoder workflow guide
│   └── private/                 # private planning and experiment notes

├── scripts/
│   ├── check_preprocessing.py         # inspect preprocessing outputs on sample data
│   ├── evaluate.py                    # evaluate a saved baseline checkpoint
│   ├── evaluate_autoencoder.py        # evaluate a saved autoencoder checkpoint
│   ├── make_splits.py                 # create reproducible data splits
│   ├── plot_autoencoder_errors.py     # plot autoencoder reconstruction errors
│   ├── plot_results.py                # generate baseline plots from saved artifacts
│   ├── train.py                       # train the baseline CNN model
│   └── train_autoencoder.py           # train the convolutional autoencoder

├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # manifest-driven dataset loader
│   │   ├── index_dataset.py     # dataset indexing helpers
│   │   ├── split.py             # split creation/loading logic
│   │   └── transforms.py        # dataset-facing preprocessing transforms
│   │
│   ├── evaluation/
│   │   ├── evaluate.py          # reusable evaluation pipeline
│   │   ├── evaluate_autoencoder.py   # autoencoder evaluation pipeline
│   │   ├── metrics.py                # metric helpers
│   │   ├── plot_autoencoder.py       # autoencoder plotting helpers
│   │   └── plots.py                  # baseline plotting helpers
│   │
│   ├── experiments/
│   │   └── run_experiment.py    # experiment entry point / orchestration
│   │
│   ├── features/
│   │   ├── spectrogram.py       # log-mel feature extraction
│   │   └── waveform.py          # waveform preprocessing utilities
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_baseline.py      # baseline CNN classifier
│   │   ├── conv_autoencoder.py  # convolutional autoencoder model
│   │   └── registry.py          # model lookup / registration helpers
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── autoencoder_trainer.py    # autoencoder training/validation loops
│   │   ├── callbacks.py              # checkpoint/history helpers
│   │   ├── losses.py                 # loss builders
│   │   ├── train.py                  # baseline training pipeline
│   │   ├── train_autoencoder.py      # autoencoder training pipeline
│   │   └── trainer.py                # shared epoch-level train/validation loops
│   │
│   └── utils/
│       ├── io.py                # path and filesystem helpers
│       └── seed.py              # reproducibility helpers
```
For a more detailed explanation of each module and its role, see:
- [Project design and architecture](docs/baseline.md)

---

## Repository Conventions

* Put reusable core logic in `src/`
* Put official runnable workflows in `scripts/`
* Save generated outputs in `artifacts/`
* Keep scripts thin; core logic should stay reusable
* Use notebooks for exploration only, not as the main pipeline
* Prefer a stable baseline before adding more experiments

For a deeper explanation of design decisions, see:
- [Baseline design documentation](docs/baseline.md)

---

## Typical Workflow

1. Place raw MIMII audio under `data/raw/`
2. Generate a saved split manifest
3. Train the baseline model on the train split and validate every epoch
4. Evaluate the saved best checkpoint on the test split
5. Generate plots from saved history and evaluation artifacts
6. Compare results across runs or machine types


---

## Current Baseline Status

The current baseline has been implemented end to end for a one-machine-type setup:

* split generation
* raw WAV dataset loading
* deterministic log-mel preprocessing
* baseline CNN training
* test-set evaluation
* visualization

The baseline currently prioritizes:

* clarity
* reproducibility
* modularity
* stable debugging over heavy experimentation

Train-time augmentation is intentionally not included yet.


---

## Experiments

This project is designed to support:

* multiple preprocessing methods
* different model architectures
* hyperparameter tuning
* comparison across machine types

The current focus is to establish a clean and stable baseline before moving to heavier experimentation such as augmentation or pretrained models.

---

## Notes

* This project prioritizes **clarity, reproducibility, and modularity**
* The codebase is structured so the baseline remains easy to inspect and extend
* Keep the saved split manifests as source-of-truth artifacts for train/validation/test membership

## Additional Experiments

This repository also includes an exploratory autoencoder-based anomaly detection approach.

The autoencoder is trained only on normal samples and uses reconstruction error as an anomaly score.

This approach was implemented and evaluated but was not adopted as the main method due to weaker separation between normal and abnormal samples under the current setup.

For usage details, see:
- [Autoencoder experiment usage](docs/autoencoder_usage.md)


## 📚 Documentation

For more details, see:

- 🔧 Setup: [docs/setup.md](docs/setup.md)
- 📊 Baseline usage: [docs/baseline_usage.md](docs/baseline_usage.md)
- 📁 Data layout: [docs/data.md](docs/data.md)
- 🧠 Baseline design: [docs/baseline.md](docs/baseline.md)
- 🔁 Reproducibility: [docs/reproducibility.md](docs/reproducibility.md)
