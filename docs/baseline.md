## Baseline Pipeline

The primary workflow in this repository is a reproducible CNN-based classification pipeline.

The current baseline path is:

1. saved split manifest
2. raw WAV dataset loader
3. deterministic preprocessing
4. CNN-based classifier
5. training loop
6. evaluation
7. visualization

This pipeline serves as the **reference implementation** for all experiments in this repository.

---

## Baseline preprocessing

The preprocessing pipeline is deterministic and runs on the fly through dataset transforms.

Current design:

* raw multichannel waveform is averaged from 8 channels to mono
* no resampling is required (dataset is consistently `16000 Hz`)
* no trimming or segmentation is applied (clips are already consistent in length)
* no additional normalization is applied in the baseline
* feature representation = **log-mel spectrogram**

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

---

## Baseline model

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

## Model variants

In addition to the baseline CNN, the repository includes controlled architecture variants:

* `baseline_cnn` — reference model used for baseline results
* `deeper_cnn` — adds additional convolutional depth
* `wider_cnn` — increases channel capacity

All models:

* share the same input/output contract
* use the same preprocessing pipeline
* plug into the same training and evaluation framework

This allows fair comparisons across model architectures without changing the data pipeline.

The baseline model remains the **default and primary reference**, while the variants are used for controlled follow-up experiments.

---

## Training configuration

The baseline training setup uses:

* mini-batch training
* Adam optimizer
* validation at each epoch
* checkpoint selection based on validation loss

Optional extensions supported in the same pipeline include:

* class imbalance handling via positive-class weighting
* early stopping
* alternative model variants

These extensions do not change the core pipeline but allow controlled experimentation around the baseline.

---

## Evaluation and metrics

Evaluation is performed on a held-out test split defined in the saved manifest.

Metrics include:

* test loss
* accuracy
* precision
* recall
* F1 score
* confusion matrix
* ROC-AUC

The default classification threshold is `0.5`.

Additional analysis (such as threshold tuning) can be performed separately without modifying the core baseline definition.

---

## Role of the baseline

The baseline pipeline is designed to be:

* reproducible
* modular
* easy to extend
* consistent across experiments

All additional experiments in this repository — including model variants, hyperparameter tuning, and the exploratory autoencoder — are built around this baseline.

It serves as the **anchor point** for comparison, interpretation, and reporting.


## From baseline to final model

The baseline pipeline serves as the starting point for all experiments in this repository.

The final reported model extends this baseline with controlled improvements:

* a deeper CNN architecture (`deeper_cnn`)
* positive-class weighting during training to address class imbalance
* threshold tuning on the validation set to improve recall

These changes are applied **without modifying the core data pipeline**, ensuring that improvements can be attributed to model and training decisions rather than changes in preprocessing or data splitting.

The baseline remains the reference configuration, while the final model represents a tuned version built on top of it.
