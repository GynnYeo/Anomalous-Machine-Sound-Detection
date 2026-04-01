# Anomalous-Machine-Sound-Detection

## Overview

This project aims to detect abnormal machine sounds using deep learning techniques on the MIMII dataset. The task is framed as a binary classification problem: `normal` vs `abnormal` audio.

The goal is not only to build a working model, but to create a **reproducible experimental pipeline** that allows comparison of:
- different preprocessing methods
- different model architectures
- different hyperparameters

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
│   ├── logs/                    # optional run logs
│   └── metrics/                 # saved training/evaluation metrics by run

├── notebooks/                   # exploration/debugging notebooks only

├── scripts/
│   ├── evaluate.py              # evaluate a saved checkpoint on the test split
│   ├── make_splits.py           # create reproducible data splits
│   ├── plot_results.py          # generate baseline plots from saved artifacts
│   └── train.py                 # train the baseline model

├── src/
│   ├── evaluation/
│   │   ├── evaluate.py          # reusable evaluation pipeline
│   │   ├── metrics.py           # evaluation metric helpers
│   │   └── plots.py             # plotting helpers
│   │
│   ├── features/
│   │   ├── spectrogram.py       # log-mel feature extraction
│   │   └── waveform.py          # waveform preprocessing utilities
│   │
│   ├── data/
│   │   ├── dataset.py           # manifest-driven dataset loader
│   │   ├── index_dataset.py     # dataset indexing helpers
│   │   ├── split.py             # split creation/loading logic
│   │   └── transforms.py        # dataset-facing transforms
│   │
│   ├── models/
│   │   └── cnn_baseline.py      # baseline CNN model
│   │
│   ├── training/
│   │   ├── callbacks.py         # checkpoint/history helpers
│   │   ├── losses.py            # baseline loss builder
│   │   ├── train.py             # reusable training pipeline
│   │   └── trainer.py           # epoch-level train/validation loops
│   │
│   └── utils/
│       ├── io.py
│       ├── logging.py
│       └── seed.py
````

---

## Data Layout

Place the raw MIMII WAV files under `data/raw/`.

Expected folder layout:

```text
data/raw/{snr_db}_{machine_type}/{machine_id}/{label}/*.wav
```

Example:

```text
data/raw/-6_dB_fan/id_00/normal/*.wav
data/raw/-6_dB_fan/id_00/abnormal/*.wav
data/raw/0_dB_pump/id_02/normal/*.wav
data/raw/6_dB_valve/id_04/abnormal/*.wav
```

Notes:

* Valid labels are `normal` and `abnormal`
* The split/index pipeline reads metadata from the folder names, so keep this structure unchanged
* `data/raw/` should contain the original audio files
* Saved split manifests are written under `data/splits/`

---

## Repository Conventions

* Put reusable core logic in `src/`
* Put official runnable workflows in `scripts/`
* Save generated outputs in `artifacts/`
* Keep scripts thin; core logic should stay reusable
* Use notebooks for exploration only, not as the main pipeline
* Prefer a stable baseline before adding more experiments

---

## Typical Workflow

1. Place raw MIMII audio under `data/raw/`
2. Generate a saved split manifest
3. Train the baseline model on the train split and validate every epoch
4. Evaluate the saved best checkpoint on the test split
5. Generate plots from saved history and evaluation artifacts
6. Compare results across runs or machine types

---

## Setup

### 1. Create and activate a virtual environment

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 2. Install PyTorch and torchaudio

Install the PyTorch stack using the official PyTorch installation selector for your platform and hardware:

* CPU-only
* CUDA
* ROCm
* Apple Silicon / MPS

After choosing the correct command for your machine, install it first.

### 3. Install the remaining project dependencies

```bash
pip install -r requirements.txt
```

### 4. Download dataset

Download the MIMII dataset and place it under:

```text
data/raw/
```

The dataset must follow:

```text
data/raw/{snr_db}_{machine_type}/{machine_id}/{label}/*.wav
```

with `label` set to `normal` or `abnormal`.

---

## Baseline Pipeline

The current baseline path is:

1. saved split manifest
2. raw WAV dataset loader
3. deterministic preprocessing
4. baseline CNN
5. training loop
6. evaluation
7. visualization

### Baseline preprocessing

The baseline preprocessing pipeline is deterministic and runs on the fly through dataset transforms.

Current baseline design:

* raw multichannel waveform is averaged from 8 channels to mono
* no resampling is needed because the dataset is already consistently `16000 Hz`
* no trim/pad/segment step is used in baseline v1 because the verified clips are already consistent in length
* no extra peak normalization is added in baseline v1
* feature representation = log-mel spectrogram

Current log-mel settings:

* `sample_rate = 16000`
* `n_fft = 1024`
* `win_length = 1024`
* `hop_length = 512`
* `n_mels = 64`
* `f_min = 0.0`
* `f_max = 8000.0`

Verified transformed model input shape per sample:

* `[1, 64, 313]`

### Baseline model

The baseline model is a CNN classifier trained on log-mel inputs.

Model contract:

* input shape: `[batch_size, 1, 64, 313]`
* output shape: `[batch_size]`
* one logit per sample for binary classification

Training uses:

* `BCEWithLogitsLoss`

Binary prediction rule during evaluation:

* `probability = sigmoid(logit)`
* predict `abnormal` if `probability >= 0.5`

---

## Usage

### 1. Generate split manifests

The split pipeline is driven by `scripts/make_splits.py`.

It:

* scans `data/raw/` and builds a master index of WAV files
* optionally saves `data/splits/master_index.csv`
* filters the dataset to a selected scope such as one machine type
* creates deterministic train/validation/test splits by group, not by individual file
* saves the resulting split manifest under `data/splits/`

For the one-machine-type baseline, each split group is defined as:

* `group_id = (machine_id, snr_db)`

This keeps all files from the same machine ID and dB condition inside exactly one split and helps reduce leakage across train, validation, and test.

Example:

```bash
python scripts/make_splits.py --machine-type fan --write-master-index
```

Useful flags:

* `--machine-type fan`
* `--write-master-index`
* `--seed 42`
* `--train-ratio`, `--val-ratio`, `--test-ratio`

Expected outputs:

* `data/splits/master_index.csv` when `--write-master-index` is used
* `data/splits/fan_split_seed42.csv` for the example above

---

### 2. Train a model

The baseline training pipeline is driven by `scripts/train.py`.

Example:

```bash
python scripts/train.py \
  --manifest-path data/splits/fan_split_seed42.csv \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 1e-3 \
  --run-name fan_baseline
```

This will:

* load the saved `train` and `val` splits from the manifest
* apply the baseline log-mel preprocessing on the fly
* train the baseline CNN using mini-batches
* validate every epoch
* save:

  * `best.pt` when validation loss improves
  * `last.pt` every epoch for resume/recovery
* save training history for later plotting

Current device priority:

1. CUDA
2. MPS
3. CPU

Example artifact layout:

```text
artifacts/
├── checkpoints/
│   └── fan_baseline/
│       ├── best.pt
│       └── last.pt
└── metrics/
    └── fan_baseline/
        ├── history.json
        └── run_config.json
```

---

### 3. Evaluate a trained model

The evaluation pipeline is driven by `scripts/evaluate.py`.

It:

* loads the saved split manifest
* creates the `test` dataset with the same baseline transform
* loads the saved best checkpoint
* runs inference on the test split
* computes baseline test metrics
* saves the results to a metrics JSON file

Example:

```bash
python scripts/evaluate.py \
  --manifest-path data/splits/fan_split_seed42.csv \
  --checkpoint-path artifacts/checkpoints/fan_baseline/best.pt \
  --run-name fan_baseline
```

Baseline evaluation metrics:

* test loss
* accuracy
* precision
* recall
* F1
* confusion matrix
* ROC-AUC

Example output artifact:

```text
artifacts/metrics/fan_baseline/test_metrics.json
```

---

### 4. Generate plots

The plotting utility is driven by `scripts/plot_results.py`.

It can generate:

* training vs validation loss curve
* validation accuracy curve
* test confusion matrix

Example:

```bash
python scripts/plot_results.py \
  --history-path artifacts/metrics/fan_baseline/history.json \
  --metrics-path artifacts/metrics/fan_baseline/test_metrics.json \
  --run-name fan_baseline
```

Example output artifacts:

```text
artifacts/curves/fan_baseline/loss_curve.png
artifacts/curves/fan_baseline/val_accuracy_curve.png
artifacts/curves/fan_baseline/confusion_matrix.png
```

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

## Reproducibility

This project supports reproducible baseline runs through:

* saved split manifests
* deterministic baseline preprocessing
* seed control
* saved checkpoints
* saved training history
* saved test metrics
* saved plots

To reproduce a run, keep:

* the manifest path
* run name
* checkpoint path
* training settings
* saved metric artifacts

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
