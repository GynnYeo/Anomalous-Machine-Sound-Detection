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

To resume training, use `--resume-from` and provide the path to a saved checkpoint, for example:
```bash 
python scripts/train.py \
  --manifest-path data/splits/fan_split_seed42.csv \
  --epochs 20 \
  --run-name fan_baseline \
  --resume-from artifacts/checkpoints/fan_baseline/last.pt
```
This resumes training from the saved checkpoint and continues the run without restarting from epoch 1.



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