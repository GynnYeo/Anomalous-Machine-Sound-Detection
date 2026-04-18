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
