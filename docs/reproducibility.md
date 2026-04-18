# Reproducibility

This project supports reproducible CNN classification runs through:

* saved split manifests
* deterministic preprocessing
* fixed random seed control
* saved checkpoints
* saved run configurations
* saved training histories
* saved evaluation metrics
* saved plots

The goal of this document is to show how to reproduce the reported results **without retraining the model from scratch**.

---

## What you need

To reproduce a reported run correctly, keep the following artifacts together:

* **Split manifest**
  Defines the exact train / validation / test split membership.

* **Saved checkpoint**
  Stores the trained model weights to be reloaded directly.

* **Training history**
  Stores per-epoch training and validation metrics for curve reproduction.

* **Saved test metrics**
  Stores the final evaluation metrics for the reported checkpoint.

* **Run configuration**
  Stores the training settings used for the saved run.

* **Selected decision threshold**
  Required if the reported result uses a tuned threshold instead of the default 0.5.

For a run named `<run_name>`, the expected files are:

```text
data/splits/<scope>_split_seed42.csv
artifacts/checkpoints/<run_name>/best.pt
artifacts/metrics/<run_name>/history.json
artifacts/metrics/<run_name>/run_config.json
artifacts/metrics/<run_name>/test_metrics.json
```

Depending on the experiment, reproducibility may also depend on:

* the selected `model_name`
* any positive-class weighting used during training
* any threshold-tuning outputs used to choose a non-default decision threshold

---

## Minimum information needed to reproduce a run

At minimum, a reproducible run should preserve:

* the manifest path
* the run name
* the checkpoint path
* the model architecture used
* the saved history JSON
* the saved test metrics JSON

If you report a tuned-threshold result rather than the default classification threshold, also keep:

* the threshold summary JSON / CSV
* the selected-threshold JSON file

---

## Reproduce evaluation metrics from a saved checkpoint

Use the evaluation script to reload a trained checkpoint and recompute the test-set metrics.

Example:

```bash
python scripts/evaluate.py \
  --manifest-path data/splits/all_machines_split_seed42.csv \
  --checkpoint-path artifacts/checkpoints/all_machines_baseline/best.pt \
  --run-name all_machines_baseline \
  --model-name baseline_cnn
```

This will:

1. load the saved split manifest
2. recreate the test split with the same preprocessing pipeline
3. reload the saved checkpoint
4. run test-set inference
5. save the final metrics JSON

Expected output artifact:

```text
artifacts/metrics/all_machines_baseline/test_metrics.json
```

If the saved run used a different architecture, replace `baseline_cnn` with the appropriate model name, for example:

* `deeper_cnn`
* `wider_cnn`

---

## Reproduce plots from saved artifacts

Use the plotting script to recreate the training curves and confusion matrix from saved JSON artifacts.

Example:

```bash
python scripts/plot_results.py \
  --history-path artifacts/metrics/all_machines_baseline/history.json \
  --metrics-path artifacts/metrics/all_machines_baseline/test_metrics.json \
  --run-name all_machines_baseline
```

This will generate:

```text
artifacts/curves/all_machines_baseline/loss_curve.png
artifacts/curves/all_machines_baseline/val_accuracy_curve.png
artifacts/curves/all_machines_baseline/confusion_matrix.png
```

These plot files are the expected source for the training and evaluation figures shown in the report.

---

## Reproduce threshold-tuning analysis (optional)

If a reported result depends on threshold tuning rather than the default `0.5` decision threshold, reproduce that step separately.

Example:

```bash
python scripts/tune_threshold.py \
  --manifest-path data/splits/all_machines_split_seed42.csv \
  --checkpoint-path artifacts/checkpoints/all_machines_baseline/best.pt \
  --metric recall \
  --model-name baseline_cnn
```

This will save:

* a threshold summary CSV
* a threshold summary JSON
* a selected-threshold JSON

If `--evaluate-test` is used, it can also save test metrics for the chosen threshold.

These artifacts should be kept together with the checkpoint if the report uses a tuned threshold.

---

## Reproduce grid-search analysis (optional)

If you want to reproduce a saved hyperparameter search summary, rerun the same grid-search configuration.

Example:

```bash
python scripts/grid_search.py \
  --manifest-path data/splits/all_machines_split_seed42.csv \
  --run-prefix all_machines_grid \
  --model-name baseline_cnn
```

This produces a ranked summary of runs and saves summary files under the grid-search artifact directory.

To recreate figures from the saved summary:

```bash
python scripts/plot_results.py \
  --grid-summary-path artifacts/grid_search/all_machines_grid_summary.json \
  --run-name all_machines_grid
```

---

## Expected baseline example results

For the current saved `all_machines_baseline` example, the reproduced test metrics should be approximately:

* accuracy: `0.9763`
* precision: `0.9887`
* recall: `0.8882`
* F1: `0.9358`
* ROC-AUC: `0.9952`

Small differences may occur across environments, but reproduced results should remain very close when using the same manifest, checkpoint, preprocessing pipeline, and model configuration.

These values correspond to the baseline run used as the current reproducibility example in this repository and should match the metrics reported from the same saved artifacts.

---

## Final reported model

The final reported model in this project builds on the baseline CNN pipeline with controlled improvements.

### Configuration summary

```text
manifest_path: data/splits/all_machines_split_seed42.csv
model_name: deeper_cnn
checkpoint_path: artifacts/checkpoints/<final_run>/best.pt
pos_weight: <value used during training>
threshold: <selected threshold from validation tuning>
run_name: <final_run_name>
```

### Reproduction steps

1. Evaluate the saved checkpoint:

```bash
python scripts/evaluate.py \
  --manifest-path data/splits/all_machines_split_seed42.csv \
  --checkpoint-path artifacts/checkpoints/<final_run>/best.pt \
  --run-name <final_run_name> \
  --model-name deeper_cnn
```

2. Regenerate plots:

```bash
python scripts/plot_results.py \
  --history-path artifacts/metrics/<final_run_name>/history.json \
  --metrics-path artifacts/metrics/<final_run_name>/test_metrics.json \
  --run-name <final_run_name>
```

3. (If applicable) Reproduce threshold tuning:

```bash
python scripts/tune_threshold.py \
  --manifest-path data/splits/all_machines_split_seed42.csv \
  --checkpoint-path artifacts/checkpoints/<final_run>/best.pt \
  --metric recall \
  --model-name deeper_cnn
```

### Notes

* The final model uses a **tuned decision threshold**, not the default 0.5.
* Positive-class weighting is applied during training to improve recall.
* All artifacts (manifest, checkpoint, metrics, threshold outputs) must be used together for full reproducibility.


---

## Reproducibility guarantees in this repository

Reproducibility in this repository relies on:

* saved split manifests as source-of-truth artifacts
* deterministic preprocessing
* fixed seeds during training
* persisted checkpoint files
* persisted run configuration files
* persisted history, metric, and plot artifacts

To reproduce a reported run correctly, always keep together:

* the manifest path
* the run name
* the checkpoint path
* the model name
* the saved history JSON
* the saved run configuration JSON
* the saved test metrics JSON

If threshold tuning was part of the reported result, also keep together:

* the threshold summary files
* the selected-threshold JSON
* any test metrics generated from the selected threshold
