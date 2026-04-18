# Reproducibility

This project supports reproducible baseline runs through:

- saved split manifests
- deterministic baseline preprocessing
- fixed random seed control
- saved checkpoints
- saved training history
- saved evaluation metrics
- saved plots

The goal of this document is to show how to reproduce the reported baseline results **without retraining the model from scratch**.

---

## What you need

To reproduce a baseline run, keep the following artifacts together:

- **Split manifest**  
  Defines the exact train / validation / test split membership.

- **Saved checkpoint**  
  Stores the trained model weights to be reloaded directly.

- **Training history**  
  Stores per-epoch training and validation metrics for curve reproduction.

- **Saved test metrics**  
  Stores the final evaluation metrics for the reported checkpoint.

For a run named `<run_name>`, the expected files are:

```text
data/splits/<scope>_split_seed42.csv
artifacts/checkpoints/<run_name>/best.pt
artifacts/metrics/<run_name>/history.json
artifacts/metrics/<run_name>/test_metrics.json
````

---

## Reproduce evaluation metrics from a saved checkpoint

Use the evaluation script to reload a trained checkpoint and compute the test-set metrics again.

Example:

```bash
python scripts/evaluate.py \
  --manifest-path data/splits/all_machines_split_seed42.csv \
  --checkpoint-path artifacts/checkpoints/all_machines_baseline/best.pt \
  --run-name all_machines_baseline
```

This will:

1. load the saved split manifest
2. recreate the test split with the baseline preprocessing
3. reload the saved model checkpoint
4. run test-set inference
5. save the final metrics JSON

Expected output artifact:

```text
artifacts/metrics/all_machines_baseline/test_metrics.json
```

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

These plot files are the expected source for the baseline training/evaluation figures shown in the report.

---

## Expected baseline example results

For the current saved `all_machines_baseline` example, the reproduced test metrics should be approximately:

* accuracy: `0.9763`
* precision: `0.9887`
* recall: `0.8882`
* F1: `0.9358`
* ROC-AUC: `0.9952`

Small differences may occur across environments, but reproduced results should remain very close when using the same manifest, checkpoint, and preprocessing pipeline.

These values correspond to the baseline run used as the current reproducibility example in this repository and should match the metrics reported from the same saved artifacts.


---

## Final model placeholder

The final best-performing version of the baseline model is maintained separately as part of the team’s final submission workflow.

Once the final checkpoint and final run metadata are available, replace the example paths above with:

```text
<FINAL_MANIFEST_PATH>
<FINAL_CHECKPOINT_PATH>
<FINAL_RUN_NAME>
```

The reproduction procedure remains the same:

1. run `scripts/evaluate.py` on the final checkpoint
2. run `scripts/plot_results.py` on the saved history and metrics
3. verify that the regenerated metrics and figures match the report

---

## Reproducibility guarantees in this repository

Baseline reproducibility in this repository relies on:

* saved split manifests as source-of-truth artifacts
* deterministic baseline preprocessing
* fixed seeds during training
* persisted checkpoint files
* persisted metric and plot artifacts

To reproduce a reported run correctly, always keep together:

* the manifest path
* the run name
* the checkpoint path
* the saved history JSON
* the saved test metrics JSON


