# Autoencoder Experiment Usage

## Purpose

This experiment explores a **reconstruction-based anomaly detection approach** using a convolutional autoencoder.

Unlike the CNN classification pipeline, this method:

* trains only on **normal samples**
* learns to reconstruct normal log-mel spectrograms
* uses **reconstruction error as an anomaly score**

This workflow is included as an **exploratory experiment**.
It is not the primary supported pipeline in this repository, but serves as an alternative approach for comparison and analysis.

---

## Data usage

The experiment uses the same saved split manifest as the baseline pipeline.

Split usage:

* **train split**: normal samples only
* **validation split**: normal + abnormal (used for threshold selection)
* **test split**: normal + abnormal (used for final evaluation)

---

## Train

```bash id="d7x3vu"
python scripts/train_autoencoder.py \
  --manifest-path data/splits/all_machines_split_seed42.csv \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --seed 42 \
  --run-name all_machines_autoencoder_v1
```

This will:

* load only normal samples from the training split
* apply the same log-mel preprocessing used in the baseline pipeline
* train the convolutional autoencoder
* save checkpoints and training history

---

## Evaluate

```bash id="v9r4lx"
python scripts/evaluate_autoencoder.py \
  --manifest-path data/splits/all_machines_split_seed42.csv \
  --checkpoint-path artifacts/checkpoints/all_machines_autoencoder_v1/best.pt \
  --batch-size 32 \
  --run-name all_machines_autoencoder_v1
```

This will:

* load the trained autoencoder checkpoint
* compute reconstruction errors on validation and test splits
* select a threshold based on validation performance
* evaluate anomaly detection performance on the test split

---

## Outputs

### Training artifacts

```text id="b4j6s1"
artifacts/checkpoints/all_machines_autoencoder_v1/best.pt
artifacts/checkpoints/all_machines_autoencoder_v1/last.pt
artifacts/metrics/all_machines_autoencoder_v1/history.json
artifacts/metrics/all_machines_autoencoder_v1/run_config.json
```

### Evaluation artifacts

```text id="u7m9xq"
artifacts/metrics/all_machines_autoencoder_v1/test_metrics.json
```

---

## Evaluation logic

The anomaly detection process follows these steps:

1. Load the best checkpoint

2. Compute reconstruction errors on the **validation split**

3. Select a threshold that maximizes validation F1

4. Compute reconstruction errors on the **test split**

5. Predict `abnormal` if:

   ```
   reconstruction_error >= threshold
   ```

6. Save final test metrics

---

## Notes

* This experiment uses the same preprocessing and data pipeline as the CNN baseline for fair comparison
* Performance depends heavily on the quality of reconstruction separation between normal and abnormal samples
* In this project, the autoencoder approach was **not pursued as the final method** due to limited anomaly separation under current constraints

This experiment is retained to:

* demonstrate an alternative anomaly-detection formulation
* support discussion in the report
* provide a foundation for future improvements
