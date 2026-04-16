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


## Autoencoder Experiment Results and Roadblock

### Experiment run

After implementing the convolutional autoencoder pipeline, the experiment was run using the all-machine split:

```bash
python scripts/train_autoencoder.py \
  --manifest-path data/splits/all_machines_split_seed42.csv \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --seed 42 \
  --run-name all_machines_autoencoder_v1
````

The trained model was then evaluated using:

```bash
python scripts/evaluate_autoencoder.py \
  --manifest-path data/splits/all_machines_split_seed42.csv \
  --checkpoint-path artifacts/checkpoints/all_machines_autoencoder_v1/best.pt \
  --batch-size 32 \
  --run-name all_machines_autoencoder_v1
```

The configuration for this experiment was:

| Component        | Setting                                      |
| ---------------- | -------------------------------------------- |
| Split            | `data/splits/all_machines_split_seed42.csv`  |
| Model            | Convolutional autoencoder                    |
| Input            | Log-mel spectrogram `[1, 64, 313]`           |
| Training data    | Normal samples from the train split only     |
| Validation data  | Normal + abnormal validation samples         |
| Test data        | Normal + abnormal test samples               |
| Loss             | `MSELoss`                                    |
| Optimizer        | Adam                                         |
| Learning rate    | `1e-3`                                       |
| Batch size       | `32`                                         |
| Epochs           | `20`                                         |
| Threshold method | Validation-selected reconstruction threshold |

### Training behavior

The autoencoder did learn to reduce reconstruction loss during training. This is important because it suggests that the training loop itself was not completely broken.

The training loss decreased substantially over the run, from a high initial reconstruction loss to a much lower final loss. The validation loss also decreased over time. This means the encoder and decoder were able to learn a reconstruction function for the log-mel spectrogram inputs.

However, reducing reconstruction loss is not the same as solving anomaly detection. For this method to work well, abnormal samples must produce noticeably higher reconstruction error than normal samples. The main issue observed in this experiment was that this separation did not happen clearly.

### Test results

The final autoencoder test results were:

| Metric    |   Result |
| --------- | -------: |
| Accuracy  | `0.1920` |
| Precision | `0.1919` |
| Recall    | `0.9836` |
| F1        | `0.3211` |
| ROC-AUC   | `0.5036` |

The confusion matrix was:

```text
TN = 8
FP = 7337
FN = 29
TP = 1742
```

This result shows that the model predicted almost every test sample as abnormal. The very high recall means that most abnormal samples were detected, but this happened because the model also incorrectly flagged almost all normal samples as abnormal.

The ROC-AUC of approximately `0.50` is the most important result. A ROC-AUC near `0.50` means the reconstruction error was almost random as an anomaly score. In other words, the model was not reliably assigning higher reconstruction error to abnormal samples than to normal samples.

### Reconstruction error analysis

To investigate the poor result, reconstruction error summaries were added for normal and abnormal samples.

Validation reconstruction error summary:

| Class    |  Count |     Mean |   Median |      p95 |      p99 |
| -------- | -----: | -------: | -------: | -------: | -------: |
| Normal   | `7374` | `5.0284` | `4.8757` | `6.9412` | `7.8878` |
| Abnormal | `1529` | `5.1399` | `4.6521` | `7.4257` | `8.6149` |

Test reconstruction error summary:

| Class    |  Count |     Mean |   Median |      p95 |      p99 |
| -------- | -----: | -------: | -------: | -------: | -------: |
| Normal   | `7345` | `5.4720` | `5.1133` | `7.8896` | `8.8137` |
| Abnormal | `1771` | `5.6371` | `5.0856` | `8.4840` | `9.5855` |

These summaries show the main roadblock clearly. Although abnormal samples had a slightly higher mean reconstruction error, the difference was very small. The medians were almost the same, and in both validation and test, the abnormal median was actually slightly lower than the normal median.

This means the reconstruction error distributions for normal and abnormal samples overlapped heavily. Therefore, there was no reliable threshold that could cleanly separate the two classes.

### Threshold experiments

The first evaluation used the validation threshold that maximized F1. This selected a very low threshold:

```text
threshold = 3.8295
```

This threshold was below the reconstruction error of most normal test samples, so almost all samples were predicted as abnormal. This led to very high recall but extremely poor precision and accuracy.

To check whether the issue was only caused by the threshold selection method, two additional threshold strategies were tested using only normal validation reconstruction errors:

| Threshold strategy    | Threshold | Accuracy | Precision |   Recall |       F1 |
| --------------------- | --------: | -------: | --------: | -------: | -------: |
| Validation best F1    |  `3.8295` | `0.1920` |  `0.1919` | `0.9836` | `0.3211` |
| Normal validation p95 |  `6.9412` | `0.7207` |  `0.2681` | `0.2530` | `0.2603` |
| Normal validation p99 |  `7.8878` | `0.7876` |  `0.3565` | `0.1158` | `0.1748` |

The percentile thresholds reduced the number of false positives, so accuracy increased. However, recall dropped heavily because many abnormal samples also had reconstruction errors within the normal range.

This confirmed that the poor result was not only caused by a bad threshold. The deeper problem was that reconstruction error itself was not a strong anomaly score in this setup.

### Why this became a roadblock

This experiment reached a roadblock because the core assumption of reconstruction-based anomaly detection did not hold strongly enough.

The assumption was:

```text
normal samples   → low reconstruction error
abnormal samples → high reconstruction error
```

However, the observed behavior was closer to:

```text
normal samples   → reconstruction error around 5
abnormal samples → reconstruction error around 5
```

Since normal and abnormal reconstruction errors were very similar, changing the threshold could only trade one type of error for another:

* low threshold: catches abnormalities but falsely flags many normal samples
* high threshold: avoids false alarms but misses most abnormalities

The ROC-AUC near `0.50` showed that no threshold would produce strong performance because the ranking of samples by reconstruction error was almost random.

### Possible reasons for the roadblock

Several factors may explain why the autoencoder did not perform well.

#### 1. Abnormal samples may still be easy to reconstruct

Even though the autoencoder was trained only on normal samples, abnormal log-mel spectrograms may still share enough structure with normal spectrograms that the decoder can reconstruct them reasonably well.

If abnormal sounds are visually similar to normal sounds in log-mel space, reconstruction error may not increase much.

#### 2. The autoencoder may generalize too broadly

The model may have learned a general reconstruction function rather than a narrow model of normality. If the encoder-decoder architecture is powerful enough, it can reconstruct many spectrogram-like inputs, including abnormal ones.

This weakens the usefulness of reconstruction error for anomaly detection.

#### 3. Mean MSE may dilute localized abnormal patterns

The anomaly score used in this experiment was the mean squared error across the entire spectrogram:

```text
reconstruction_error = mean((input - reconstruction)^2)
```

If abnormal information appears only in a small region of the time-frequency representation, averaging over the whole spectrogram may dilute the signal. Most of the spectrogram may still reconstruct well, causing the final average error to remain similar to normal samples.

#### 4. The all-machine setup may be too broad

This experiment trained one autoencoder across all machine types. Normal sounds from fans, pumps, sliders, and valves can be very different from each other. As a result, the autoencoder may have learned a broad normal reconstruction space.

A broader normal space makes it harder for abnormal samples to appear unusual based only on reconstruction error.

#### 5. Reconstruction quality is not the same as classification usefulness

The supervised CNN baseline learns directly from class labels, so it can focus on features that separate normal and abnormal samples. The autoencoder only learns to reconstruct inputs. A feature that is useful for classification may not necessarily produce a large reconstruction error.

This explains why the supervised CNN can perform strongly while the autoencoder performs poorly.

### Decision to stop this branch

At this stage, the autoencoder branch was stopped instead of continuing with more variations.

This was a deliberate decision. The goal of the autoencoder experiment was to test whether a simple reconstruction-based anomaly detection approach would be promising under the current pipeline. The first result showed that the basic version did not produce a useful anomaly score.

Possible future improvements could include:

* training separate autoencoders per machine type
* using a smaller bottleneck
* trying `L1Loss` instead of `MSELoss`
* using top-k reconstruction error instead of mean reconstruction error
* using denoising autoencoders
* using a stricter anomaly detection setup

However, these would each introduce additional experimental branches. Since the current project already has a strong supervised CNN baseline, and since the first autoencoder result showed near-random anomaly ranking, continuing this branch was considered lower priority.

### Takeaway

The autoencoder experiment was useful even though the result was poor.

It showed that a basic convolutional autoencoder trained on normal log-mel spectrograms was able to learn reconstruction, but reconstruction error did not meaningfully separate normal and abnormal machine sounds in the all-machine setting.

The main takeaway is:

```text
The autoencoder did not fail because it could not reconstruct.
It failed because reconstruction error was not discriminative enough for anomaly detection.
```

Compared with the supervised CNN baseline, the autoencoder was much weaker. This suggests that, for the current split, preprocessing, and dataset setup, supervised classification is a more effective approach than simple reconstruction-based anomaly detection.

This result is still valuable because it documents a tested alternative method and explains why the project should not continue investing in this direction without a more substantial redesign of the autoencoder approach.

