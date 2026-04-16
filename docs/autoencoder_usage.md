# Autoencoder Experiment Usage

## Purpose

This experiment trains a convolutional autoencoder for reconstruction-based anomaly detection.

Unlike the CNN baseline, the autoencoder is trained only on normal training samples. It learns to reconstruct normal log-mel spectrograms. During evaluation, reconstruction error is used as the anomaly score.

## Data usage

- train split: normal samples only
- validation split: normal + abnormal, used to choose threshold
- test split: normal + abnormal, used for final evaluation

## Train

```bash
python scripts/train_autoencoder.py \
  --manifest-path data/splits/all_machines_split_seed42.csv \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --seed 42 \
  --run-name all_machines_autoencoder_v1
````

## Evaluate

```bash
python scripts/evaluate_autoencoder.py \
  --manifest-path data/splits/all_machines_split_seed42.csv \
  --checkpoint-path artifacts/checkpoints/all_machines_autoencoder_v1/best.pt \
  --batch-size 32 \
  --run-name all_machines_autoencoder_v1
```

## Outputs

Training saves:

```text
artifacts/checkpoints/all_machines_autoencoder_v1/best.pt
artifacts/checkpoints/all_machines_autoencoder_v1/last.pt
artifacts/metrics/all_machines_autoencoder_v1/history.json
artifacts/metrics/all_machines_autoencoder_v1/run_config.json
```

Evaluation saves:

```text
artifacts/metrics/all_machines_autoencoder_v1/test_metrics.json
```

## Evaluation logic

1. Load the best checkpoint.
2. Compute reconstruction errors on the validation split.
3. Select the threshold that maximizes validation F1.
4. Compute reconstruction errors on the test split.
5. Predict abnormal when reconstruction error is greater than or equal to the selected threshold.
6. Save final test metrics.

````

Then in your README, add a small section like:

```markdown
## Detailed Usage Guides

For detailed reproducibility instructions, see:

- [Baseline CNN usage](docs/baseline_usage.md)
- [Autoencoder experiment usage](docs/autoencoder_usage.md)
