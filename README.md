# Anomalous-Machine-Sound-Detection

## Overview

This project detects abnormal machine sounds from the MIMII dataset using deep learning in PyTorch.

The main task is framed as binary classification:

- `normal`
- `abnormal`

The repository is designed around a reproducible workflow:

- generate deterministic split manifests
- train a CNN classifier from scratch
- reload a saved checkpoint
- reproduce evaluation metrics and figures used in the report

The primary supported path in this repository is the CNN classification pipeline.  
Additional model variants and tuning utilities are included for controlled follow-up experiments.  
An exploratory autoencoder branch is also included as a secondary experiment.

For a deeper explanation of the baseline pipeline and design decisions, see:

- [Baseline design](docs/baseline.md)
---

## Final Reported Model (Summary)

The strongest-performing model in this project extends the baseline CNN pipeline with controlled improvements:

- architecture: `deeper_cnn`
- class imbalance handling: positive-class weighting (`pos_weight`)
- decision threshold: tuned on validation set (not fixed at 0.5)

This model was selected based on improved recall while maintaining high precision.

> Note: Exact configuration (manifest, checkpoint, and threshold) should be used together for full reproducibility as described in the reproducibility guide.

---

## Setup

Before running the project, follow:

- [Setup guide](docs/setup.md)

This includes:

- Python environment setup
- PyTorch / TorchAudio installation notes
- dataset placement
- troubleshooting notes for audio backends

---

## Quick Start

This is the recommended end-to-end workflow for the main CNN pipeline.

For a more detailed walkthrough of the baseline workflow, see:

- [Baseline usage guide](docs/baseline_usage.md)

### 1. Generate a split manifest

```bash
python scripts/make_splits.py --machine-type fan --seed 42
````

This creates a deterministic manifest under `data/splits/`.

For details on dataset layout and group-based splitting, see:

* [Data and split design](docs/data.md)

### 2. Train a CNN model

```bash
python scripts/train.py \
  --manifest-path data/splits/fan_split_seed42.csv \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 1e-3 \
  --run-name fan_baseline \
  --model-name baseline_cnn
```

The default architecture is `baseline_cnn`.

Other supported CNN variants are:

* `deeper_cnn`
* `wider_cnn`

Example:

```bash
python scripts/train.py \
  --manifest-path data/splits/fan_split_seed42.csv \
  --epochs 10 \
  --run-name fan_deeper \
  --model-name deeper_cnn
```

### 3. Evaluate a saved checkpoint

```bash
python scripts/evaluate.py \
  --manifest-path data/splits/fan_split_seed42.csv \
  --checkpoint-path artifacts/checkpoints/fan_baseline/best.pt \
  --run-name fan_baseline \
  --model-name baseline_cnn
```

This evaluates the saved checkpoint on the test split and writes `test_metrics.json`.

### 4. Generate plots

```bash
python scripts/plot_results.py \
  --history-path artifacts/metrics/fan_baseline/history.json \
  --metrics-path artifacts/metrics/fan_baseline/test_metrics.json \
  --run-name fan_baseline
```

This generates:

* training / validation loss curve
* validation accuracy curve
* confusion matrix

### Expected outputs

```text
artifacts/checkpoints/fan_baseline/best.pt
artifacts/checkpoints/fan_baseline/last.pt
artifacts/metrics/fan_baseline/history.json
artifacts/metrics/fan_baseline/run_config.json
artifacts/metrics/fan_baseline/test_metrics.json
artifacts/curves/fan_baseline/loss_curve.png
artifacts/curves/fan_baseline/val_accuracy_curve.png
artifacts/curves/fan_baseline/confusion_matrix.png
```

---

## Reproduce a Saved Result

To reproduce reported results without retraining, run evaluation on a saved checkpoint and then regenerate the plots.

Example:

```bash
python scripts/evaluate.py \
  --manifest-path data/splits/all_machines_split_seed42.csv \
  --checkpoint-path artifacts/checkpoints/all_machines_baseline/best.pt \
  --run-name all_machines_baseline \
  --model-name baseline_cnn
```

Then regenerate figures:

```bash
python scripts/plot_results.py \
  --history-path artifacts/metrics/all_machines_baseline/history.json \
  --metrics-path artifacts/metrics/all_machines_baseline/test_metrics.json \
  --run-name all_machines_baseline
```

For a more detailed reproducibility walkthrough, see:

* [Reproducibility guide](docs/reproducibility.md)

---

## Main Features

* PyTorch-based training and evaluation pipeline
* Deterministic manifest-driven dataset splits
* Raw WAV loading from saved manifests
* Deterministic on-the-fly log-mel preprocessing
* Multiple CNN architectures through a shared training pipeline
* Saved checkpoints, run configs, histories, metrics, and plots
* Reproducible reload-and-evaluate workflow
* Utilities for grid search and threshold tuning
* Exploratory autoencoder branch for reconstruction-based anomaly detection

---

## Main Workflow

The main supported workflow in this repository is:

1. create a saved split manifest
2. train a CNN classifier
3. select the best checkpoint by validation loss
4. evaluate on the test split
5. regenerate metrics and plots from saved artifacts

The baseline pipeline prioritizes:

* clarity
* reproducibility
* modularity
* controlled experimentation

---

### Task framing

Although anomaly detection is often approached as an unsupervised problem, this repository adopts a **supervised binary classification formulation** using labeled normal and abnormal samples.

This allows:
- direct optimization of classification metrics (precision, recall, F1)
- controlled comparison across model variants
- simpler reproducibility and evaluation

---

## Extended CNN Experiments

In addition to the baseline classifier, the repository includes controlled extensions for follow-up experiments.

### Model variants

The available model names are:

* `baseline_cnn`
* `deeper_cnn`
* `wider_cnn`

These use the same input contract and fit into the same train/evaluate pipeline.

### Class imbalance handling

Training also supports positive-class weighting:

* manual weighting with `--pos-weight`
* automatic weighting from the train split with `--auto-pos-weight`

Example:

```bash
python scripts/train.py \
  --manifest-path data/splits/all_machines_split_seed42.csv \
  --epochs 10 \
  --run-name all_machines_weighted \
  --model-name baseline_cnn \
  --auto-pos-weight
```

### Early stopping

Optional early stopping is supported:

```bash
python scripts/train.py \
  --manifest-path data/splits/all_machines_split_seed42.csv \
  --epochs 30 \
  --run-name all_machines_earlystop \
  --model-name baseline_cnn \
  --early-stopping-patience 5 \
  --early-stopping-min-delta 0.001
```

### Grid search

A small grid-search utility is included for controlled hyperparameter sweeps.

Example:

```bash
python scripts/grid_search.py \
  --manifest-path data/splits/all_machines_split_seed42.csv \
  --run-prefix all_machines_grid \
  --model-name baseline_cnn
```

This trains and evaluates multiple runs, then saves a ranked summary.

### Threshold tuning

A threshold-sweep utility is also included to study the tradeoff between precision, recall, and F1 for a saved checkpoint.

Example:

```bash
python scripts/tune_threshold.py \
  --manifest-path data/splits/all_machines_split_seed42.csv \
  --checkpoint-path artifacts/checkpoints/all_machines_baseline/best.pt \
  --metric recall \
  --model-name baseline_cnn
```

The plotting script can also generate:

* grid-search summary plots
* threshold tradeoff plots

---

## Exploratory Autoencoder Experiment

This repository also includes an exploratory convolutional autoencoder workflow for reconstruction-based anomaly detection.

This branch was implemented to test an alternative anomaly-detection framing:

* train on normal samples only
* reconstruct log-mel spectrograms
* use reconstruction error as anomaly score

It is included as a documented secondary experiment, but it is not the primary supported workflow for this repository.

For usage details, see:

* [Autoencoder experiment usage](docs/autoencoder_usage.md)

---

## Documentation Map

* [Setup guide](docs/setup.md)
* [Data and split design](docs/data.md)
* [Baseline design](docs/baseline.md)
* [Baseline usage](docs/baseline_usage.md)
* [Reproducibility guide](docs/reproducibility.md)
* [Autoencoder usage](docs/autoencoder_usage.md)

---

## Notebook Walkthrough

A Jupyter notebook is included for a guided, visual walkthrough of the project.

It is intended to complement the script-based pipeline by showing:

- dataset and split overview
- preprocessing and log-mel spectrogram visualization
- model overview
- loading saved artifacts
- evaluation metrics and plots

Notebook:

- `project_walkthrough.ipynb`

---

## Project Structure

```text
project_root/
├── scripts/
│   ├── check_preprocessing.py         # inspect waveform/log-mel preprocessing outputs on sample data
│   ├── evaluate.py                    # evaluate a saved CNN checkpoint on the test split
│   ├── evaluate_autoencoder.py        # evaluate a saved autoencoder checkpoint using reconstruction error
│   ├── grid_search.py                 # run small hyperparameter sweeps across training settings
│   ├── make_splits.py                 # build deterministic train/val/test split manifests from raw data
│   ├── plot_autoencoder_errors.py     # plot reconstruction-error distributions for the autoencoder experiment
│   ├── plot_results.py                # generate training, evaluation, grid-search, and threshold-tuning plots
│   ├── train.py                       # train a selected CNN classifier from scratch
│   ├── train_autoencoder.py           # train the convolutional autoencoder on normal samples only
│   └── tune_threshold.py              # sweep decision thresholds for a saved CNN checkpoint
│
├── src/
│   ├── data/
│   │   ├── dataset.py                 # manifest-driven PyTorch dataset for loading raw WAV samples and labels
│   │   ├── index_dataset.py           # scan raw data folders and build a master file index
│   │   ├── split.py                   # create, save, load, and summarize deterministic split manifests
│   │   └── transforms.py              # dataset-facing preprocessing transforms for model inputs
│   │
│   ├── evaluation/
│   │   ├── evaluate.py                # reusable CNN evaluation pipeline for saved checkpoints
│   │   ├── evaluate_autoencoder.py    # reusable autoencoder evaluation pipeline with anomaly scoring
│   │   ├── metrics.py                 # binary-classification metric helpers such as accuracy, precision, recall, F1, ROC-AUC
│   │   ├── plot_autoencoder.py        # plotting helpers for autoencoder reconstruction-error analysis
│   │   └── plots.py                   # plotting helpers for curves, confusion matrix, grid search, and threshold tuning
│   │
│   ├── experiments/
│   │   └── run_experiment.py          # experiment orchestration helper for custom multi-step runs
│   │
│   ├── features/
│   │   ├── spectrogram.py             # low-level spectrogram and log-mel feature extraction utilities
│   │   └── waveform.py                # waveform preprocessing helpers such as channel handling
│   │
│   ├── models/
│   │   ├── cnn_baseline.py            # reference CNN classifier for log-mel binary classification
│   │   ├── cnn_deeper.py              # deeper CNN variant with an additional convolution block
│   │   ├── cnn_wider.py               # wider CNN variant with increased channel capacity
│   │   ├── conv_autoencoder.py        # convolutional autoencoder for reconstruction-based anomaly detection
│   │   └── registry.py                # model factory for selecting architectures by name
│   │
│   ├── training/
│   │   ├── autoencoder_trainer.py     # epoch-level training and validation loops for the autoencoder
│   │   ├── callbacks.py               # checkpoint, history, and JSON save/load helpers
│   │   ├── losses.py                  # loss builders, including weighted BCE options for classifier training
│   │   ├── train.py                   # high-level CNN training orchestration with checkpointing and early stopping
│   │   ├── train_autoencoder.py       # high-level autoencoder training orchestration
│   │   └── trainer.py                 # shared epoch-level train/validation routines for CNN classifiers
│   │
│   └── utils/
│       ├── io.py                      # portable path-resolution helpers for repo-relative artifact handling
│       └── seed.py                    # random seed setup for reproducible experiments
│
├── project_walkthrough.ipynb      # guided notebook walkthrough of the project and results
```

---

## Repository Conventions

* reusable core logic lives in `src/`
* thin runnable entry points live in `scripts/`
* generated outputs are written under `artifacts/`
* saved split manifests are treated as source-of-truth artifacts
* the main pipeline is script-driven, not notebook-driven
* notebooks should support explanation and visualization, not replace the core training/evaluation code

---

## Limitations and Trade-offs

- The model tends to be conservative, favoring high precision over recall.
- Threshold selection significantly affects performance and must be tuned depending on application requirements.
- Results are based on a supervised formulation and are not directly comparable to unsupervised anomaly-detection benchmarks.

These trade-offs are discussed in more detail in the project report.

---

## Notes

* The main intended workflow is the CNN classification pipeline.
* The repository is structured to support fair comparisons across model variants and training settings.
* The autoencoder branch is retained as an exploratory experiment rather than the primary final method.
* Final report figures should always be reproducible from saved artifacts.

