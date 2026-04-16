## Autoencoder-Based Anomaly Detection Experiment

After establishing the supervised CNN baseline, an additional experimental branch was introduced using an autoencoder. The purpose of this experiment was not to replace the baseline CNN, but to compare the supervised classification approach against a reconstruction-based anomaly detection approach.

The motivation for trying an autoencoder is that abnormal machine sound detection naturally relates to anomaly detection. Instead of directly learning a boundary between `normal` and `abnormal` samples, an autoencoder can be trained to learn the structure of normal machine sounds. If the model learns to reconstruct normal samples well, then abnormal samples may produce larger reconstruction errors and can be detected as anomalies.

### Difference from the baseline CNN

The baseline CNN is a supervised binary classifier. It is trained using both normal and abnormal samples, and it learns to output one logit per sample. During evaluation, the logit is converted into a probability using a sigmoid function, and a threshold is applied to classify the sample as normal or abnormal.

The autoencoder works differently. It does not directly output a class prediction. Instead, it receives a log-mel spectrogram as input and tries to reconstruct the same spectrogram as output.

The training objective is therefore:

```text
input spectrogram → encoder → latent representation → decoder → reconstructed spectrogram
````

The model is optimized so that the reconstructed spectrogram is as close as possible to the original input spectrogram.

### Why train only on normal samples

For this experiment, the autoencoder is trained only on normal samples from the training split.

This is an important design choice. Since the model only sees normal examples during training, it should become specialized at reconstructing normal machine sound patterns. During evaluation, both normal and abnormal samples are passed through the autoencoder. The assumption is that normal samples will be reconstructed well, while abnormal samples will be reconstructed less accurately.

This means the anomaly score is based on reconstruction error:

```text
low reconstruction error  → likely normal
high reconstruction error → likely abnormal
```

In this project, the saved split manifest remains the source of truth. A new `label_filter` option was added to the dataset loader so that the autoencoder training dataset can use:

```text
split == train AND label == normal
```

Validation and test datasets are not filtered, because they must contain both normal and abnormal samples for threshold selection and final evaluation.

The resulting data usage is:

```text
train split      → normal samples only
validation split → normal + abnormal samples
test split       → normal + abnormal samples
```

### Reusing the existing preprocessing pipeline

The autoencoder experiment reuses the same baseline preprocessing pipeline as the CNN baseline. Raw 8-channel audio is averaged to mono and converted into a log-mel spectrogram.

This keeps the experiment controlled because the input representation remains unchanged. The main experimental change is the learning method, not the data split or preprocessing.

The input shape remains:

```text
[1, 64, 313]
```

When batched, the model receives:

```text
[batch_size, 1, 64, 313]
```

This is the same input shape used by the baseline CNN.

### Autoencoder architecture

A convolutional autoencoder was used because log-mel spectrograms are two-dimensional time-frequency representations. Since local patterns in spectrograms can appear across nearby time and frequency regions, convolutional layers are a natural choice.

The model consists of two main parts:

1. an encoder
2. a decoder

The encoder compresses the input spectrogram into a smaller latent representation. The decoder then attempts to reconstruct the original spectrogram from this compressed representation.

At a high level, the architecture follows this pattern:

```text
Input: [B, 1, 64, 313]

Encoder:
Conv2d → ReLU → MaxPool
Conv2d → ReLU → MaxPool
Conv2d → ReLU → MaxPool

Decoder:
ConvTranspose2d → ReLU
ConvTranspose2d → ReLU
ConvTranspose2d

Output: [B, 1, 64, 313]
```

Because the time dimension of the input spectrogram is `313`, repeated pooling and upsampling may not naturally return exactly the original shape. To keep the first autoencoder experiment simple and stable, the reconstructed output is resized back to the original input height and width before computing the loss.

### Training objective

Unlike the CNN baseline, the autoencoder does not use `BCEWithLogitsLoss`, because it is not directly predicting class labels.

Instead, it uses mean squared error loss:

```text
MSE(original spectrogram, reconstructed spectrogram)
```

The goal is to minimize the pixel-wise difference between the input spectrogram and its reconstruction.

For each training batch:

```text
input spectrogram → autoencoder → reconstruction
loss = MSE(reconstruction, input spectrogram)
```

Only normal samples are used during training, so the model learns to reconstruct normal patterns.

### Validation and threshold selection

After training, the autoencoder still does not directly output `normal` or `abnormal`. A threshold must be chosen to convert reconstruction error into a binary prediction.

For each validation sample, the reconstruction error is computed as:

```text
reconstruction_error = mean((input - reconstruction)^2)
```

The validation split contains both normal and abnormal samples. This allows different candidate thresholds to be tested. A simple first strategy is to choose the threshold that gives the best validation F1 score.

The prediction rule is:

```text
if reconstruction_error >= threshold:
    predict abnormal
else:
    predict normal
```

The threshold is selected using validation data only. The test set is not used for threshold selection, because that would make the final evaluation unfair.

### Test evaluation

Once the threshold has been selected from the validation set, the autoencoder is evaluated on the test split.

The test procedure is:

1. load the saved best autoencoder checkpoint
2. pass each test spectrogram through the autoencoder
3. compute reconstruction error for each test sample
4. classify each sample using the validation-selected threshold
5. compute final test metrics

The same metrics as the CNN baseline are used where possible:

* accuracy
* precision
* recall
* F1 score
* confusion matrix
* ROC-AUC

For ROC-AUC, the reconstruction error is treated as the anomaly score. Higher reconstruction error means the sample is considered more likely to be abnormal.

### Fair comparison against the CNN baseline

This experiment is compared against the existing all-machine CNN baseline. To keep the comparison fair, the following components are kept fixed:

* same saved split manifest
* same train/validation/test membership
* same log-mel preprocessing
* same input shape
* same evaluation metrics
* same test split for final reporting

The main differences are:

| Component       | CNN baseline                         | Autoencoder experiment                       |
| --------------- | ------------------------------------ | -------------------------------------------- |
| Training labels | normal + abnormal                    | normal only                                  |
| Objective       | binary classification                | reconstruction                               |
| Loss            | BCEWithLogitsLoss                    | MSELoss                                      |
| Model output    | abnormal logit                       | reconstructed spectrogram                    |
| Anomaly score   | sigmoid probability                  | reconstruction error                         |
| Threshold       | fixed or tuned probability threshold | validation-selected reconstruction threshold |

This makes the autoencoder a different method family rather than a small modification of the CNN.

### Expected strengths

The autoencoder approach is useful because it tests whether normal machine sounds have a consistent structure that can be learned without using abnormal labels during training.

This may be valuable in real anomaly detection settings where abnormal samples are rare, expensive to collect, or incomplete. If an autoencoder can detect abnormal sounds using only normal training data, it provides a more anomaly-focused alternative to supervised classification.

### Expected weaknesses

There are also important limitations.

First, an autoencoder may reconstruct abnormal samples well if they are visually similar to normal samples in log-mel space. In that case, the reconstruction error may not separate the two classes clearly.

Second, if the autoencoder is too powerful, it may learn to reconstruct almost any input, including abnormal samples. This would reduce its usefulness for anomaly detection.

Third, reconstruction quality does not always align perfectly with classification usefulness. A sample can have a low reconstruction error but still contain abnormal characteristics that matter for detection.

For these reasons, the goal of this experiment is not necessarily to beat the supervised CNN baseline. Instead, the goal is to understand whether reconstruction-based anomaly detection is a useful alternative for this dataset.

### Summary

The autoencoder experiment extends the project from supervised binary classification into reconstruction-based anomaly detection. The model is trained only on normal training samples and learns to reconstruct log-mel spectrograms. During evaluation, reconstruction error is used as an anomaly score. A threshold is selected using the validation split, and final performance is reported on the test split.

This experiment keeps the existing data split and preprocessing pipeline fixed, making it a controlled comparison against the established CNN baseline. Whether or not it outperforms the CNN, the autoencoder branch provides useful insight into how well normal-only reconstruction can separate abnormal machine sounds in the MIMII dataset.

```
```
