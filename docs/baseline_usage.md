## Baseline Usage

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

The training pipeline is driven by `scripts/train.py`.

Example:

```bash
python scripts/train.py \
  --manifest-path data/splits/fan_split_seed42.csv \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 1e-3 \
  --run-name fan_baseline \
  --model-name baseline_cnn
```

This will:

* load the saved `train` and `val` splits from the manifest
* apply log-mel preprocessing on the fly
* train the selected CNN model using mini-batches
* validate every epoch
* save:

  * `best.pt` when validation loss improves
  * `last.pt` every epoch for resume/recovery
* save training history for later plotting

#### Model selection

You can select different CNN architectures using `--model-name`:

* `baseline_cnn` (default)
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

#### Handling class imbalance

You can adjust the positive class weight:

* Manual weighting:

```bash
--pos-weight 2.0
```

* Automatic weighting (computed from training split):

```bash
--auto-pos-weight
```

This modifies the binary cross-entropy loss used during training.

#### Early stopping (optional)

You can enable early stopping:

```bash
python scripts/train.py \
  --manifest-path data/splits/fan_split_seed42.csv \
  --epochs 30 \
  --run-name fan_earlystop \
  --early-stopping-patience 5 \
  --early-stopping-min-delta 0.001
```

#### Resume training

To resume training:

```bash
python scripts/train.py \
  --manifest-path data/splits/fan_split_seed42.csv \
  --epochs 20 \
  --run-name fan_baseline \
  --resume-from artifacts/checkpoints/fan_baseline/last.pt
```

This continues training from the saved checkpoint without restarting.

#### Device priority

Training uses:

1. CUDA
2. MPS
3. CPU

#### Example artifact layout

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
* creates the `test` dataset with the same preprocessing
* loads the saved checkpoint
* runs inference on the test split
* computes evaluation metrics
* saves the results to a metrics JSON file

Example:

```bash
python scripts/evaluate.py \
  --manifest-path data/splits/fan_split_seed42.csv \
  --checkpoint-path artifacts/checkpoints/fan_baseline/best.pt \
  --run-name fan_baseline \
  --model-name baseline_cnn
```

Evaluation metrics:

* test loss
* accuracy
* precision
* recall
* F1
* confusion matrix
* ROC-AUC

Example output:

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

Example outputs:

```text
artifacts/curves/fan_baseline/loss_curve.png
artifacts/curves/fan_baseline/val_accuracy_curve.png
artifacts/curves/fan_baseline/confusion_matrix.png
```

---

### 5. Extended experiments (optional)

The repository also includes utilities for controlled experimentation beyond the baseline workflow.

#### Grid search

Run a small hyperparameter sweep:

```bash
python scripts/grid_search.py \
  --manifest-path data/splits/fan_split_seed42.csv \
  --run-prefix fan_grid \
  --model-name baseline_cnn
```

This will:

* train multiple configurations
* evaluate each run
* save a ranked summary

#### Threshold tuning

Tune classification threshold for a saved checkpoint:

```bash
python scripts/tune_threshold.py \
  --manifest-path data/splits/fan_split_seed42.csv \
  --checkpoint-path artifacts/checkpoints/fan_baseline/best.pt \
  --metric recall \
  --model-name baseline_cnn
```

This helps explore trade-offs between:

* precision
* recall
* F1

Additional plots supported:

* grid search comparison plots
* threshold trade-off curves

---

### Summary

The baseline pipeline supports:

* deterministic data splitting
* CNN-based training and evaluation
* reproducible saved artifacts
* multiple model variants
* class imbalance handling
* early stopping
* optional hyperparameter search
* optional threshold tuning

The recommended workflow remains:

1. generate a split
2. train a model
3. evaluate the best checkpoint
4. reproduce plots

