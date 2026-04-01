## Data Splitting Strategy

A key design decision in this project was the choice of train/validation/test splitting strategy. Since the overall project emphasizes a **reproducible experimental pipeline**, fixed splits were treated as a first-class artifact rather than something regenerated during every run. This supports the broader goal of keeping experiments controlled, comparable, and easy to reproduce.

### Why the split design matters

The task in this project is binary classification of machine audio into **normal** and **abnormal** classes. At first glance, it may seem sufficient to randomly split all audio files into training, validation, and test sets. However, this can create a **data leakage** problem.

In this dataset, audio files are not fully independent from one another. Multiple files may come from the same machine type, the same machine ID, and the same dB condition. Even when the audio clips are not identical, they may still share strong underlying similarities because they come from the same source conditions.

If highly related samples appear in both training and test sets, the model may achieve overly optimistic performance. Instead of learning general patterns of abnormality, it may partly learn source-specific characteristics such as the sound signature of a particular machine instance or recording condition. This weakens the validity of the test set, because the evaluation no longer reflects meaningfully held-out data.

### Data leakage and high correlation

In this dataset, each machine type is recorded under several dB conditions and multiple machine IDs. For example, the repository structure includes fan, pump, slider, and valve recordings across different dB settings and machine IDs such as `id_00`, `id_02`, `id_04`, and `id_06`.

Even when two audio clips are different files, clips from the same machine ID and the same dB condition are likely to remain highly correlated because they share:

- the same machine instance
- the same recording setup or environment
- the same background noise condition
- similar operating characteristics

Because of this, placing some of these files in training and others in testing can make the task artificially easier. The model may appear to generalize well when it is actually benefiting from repeated source-specific patterns.

### Choice of grouping unit

To reduce this risk, the split was not performed at the individual file level. Instead, related samples were grouped first, and the split was performed at the **group level**.

For the baseline fan experiment, the grouping unit was defined as:

`group = (machine_id, snr_db)`

This means that all files belonging to the same machine ID under the same dB condition were kept together in a single split. For example, if the selected scope was the `fan` machine type, then combinations such as:

- `(id_00, -6 dB)`
- `(id_00, 0 dB)`
- `(id_00, 6 dB)`
- `(id_02, -6 dB)`

were treated as separate groups.

Since the fan subset contains 4 machine IDs and 3 dB conditions, this produces 12 groups in total. These groups, rather than individual WAV files, were then assigned to train, validation, and test sets.

### Why this grouping was chosen

This grouping strategy was chosen as a balance between two goals:

1. **Reduce leakage by keeping highly related samples together**
2. **Retain enough groups to form practical train/validation/test splits**

A stricter alternative would have been to split only by `machine_id`, meaning that all dB conditions for one machine ID would be kept in the same split. While this gives stronger separation, it also creates very few split units. For the fan subset, splitting only by machine ID would give only 4 units, which is quite coarse for creating train, validation, and test partitions.

By using `(machine_id, snr_db)` as the grouping rule instead, the split still respects important source relationships while providing more flexibility in forming practical splits.

### Class balance considerations

The dataset is still a binary classification problem, so preserving a reasonable normal/abnormal distribution across train, validation, and test sets remains important. However, exact class-level 70/15/15 splitting was not treated as the top priority if it conflicted with group integrity.

For the baseline, a simple group-based split was preferred over a more complex stratified grouping procedure. The priority order was:

1. keep related samples in the same split
2. reduce leakage
3. preserve class proportions as closely as practical

This means the final class ratios in each split may be approximate rather than perfectly identical to the global dataset ratio. This trade-off was accepted because a slightly imperfect class balance is usually less harmful than leakage between training and test data.

### Filtering before splitting

Another important design choice was to **filter the dataset scope first, then split**. For example, if the baseline experiment focuses only on the `fan` machine type, the relevant fan rows are first selected from the master dataset index, and then the group-based split is applied only to that subset.

This allows different experiment scopes to maintain their own saved split manifests. For example:

- a `fan`-only baseline can use one fixed split file
- an all-machine experiment can use a different fixed split file

This keeps the split aligned with the scope of the experiment being run.

### Multi-machine scope

For a single-machine baseline such as `fan`, the grouping rule is:

`group = (machine_id, snr_db)`

This keeps highly related samples together while still providing enough groups to form practical train, validation, and test splits.

For an all-machine experiment, the split design can be extended so that each machine type is first split separately by groups and the resulting train, validation, and test partitions are then concatenated. This keeps the group-based leakage protection while helping preserve a more balanced representation of fan, pump, slider, and valve data across the final splits.

### Reproducibility and saved split manifests

To support reproducibility, the split was generated once using a fixed random seed and then saved as a CSV manifest under `data/splits/`.

The split manifest stores one row per file together with metadata such as:

- file path
- machine type
- dB condition
- machine ID
- class label
- group ID
- split assignment

Saving the split explicitly is preferable to recomputing it every run, because it guarantees that all later experiments use exactly the same data partition. This keeps baseline comparisons fair and avoids accidental changes caused by code edits, file ordering differences, or future dataset filtering changes.

### Summary

In summary, the data was split using a **group-based strategy** rather than a file-level random split. This choice was made to reduce data leakage caused by highly correlated samples from the same machine ID and dB condition, while still preserving enough variation for effective training. The selected grouping rule, `(machine_id, snr_db)`, provided a practical balance between evaluation reliability and split flexibility. This design also aligns with the broader project goals of clarity, modularity, controlled experimentation, and full reproducibility.

---


## Dataset Loading Pipeline

After fixing the train/validation/test split strategy, the next step was to implement the dataset loader. This module serves as the bridge between the saved split manifest and the later preprocessing and training stages.

The purpose of `src/data/dataset.py` is not to redesign the split or perform model-specific processing. Instead, its role is to take the saved split file as the source of truth, expose one selected split as a PyTorch dataset, and load one raw audio sample at a time in a consistent format.

### Why a dedicated dataset loader is needed

Once the split manifest has been created, the project needs a reliable way to access the correct files for training, validation, and testing. Although the split CSV already records which files belong to each subset, the model cannot use the CSV directly. The information in the manifest must still be translated into actual audio tensors and target labels.

This is the purpose of the dataset loader. It ensures that the rest of the pipeline does not need to manually handle file paths, split filtering, or label conversion every time data is accessed. Instead, these responsibilities are centralized in one reusable module.

This supports the broader project goals of:

- reproducibility, because the saved split manifest remains the single source of truth
- modularity, because data loading is separated from preprocessing and training
- clarity, because one module owns the logic for turning manifest rows into usable samples

### Core responsibility of `dataset.py`

The dataset loader was designed with a deliberately narrow responsibility. Its job is to:

1. read a saved split manifest
2. validate that the manifest contains the expected columns
3. filter rows by a selected split such as `train`, `val`, or `test`
4. lazily load the referenced WAV file when a sample is requested
5. convert the class label into a binary target
6. return the sample in a consistent structure that can later support preprocessing

Just as importantly, the dataset loader was intentionally **not** made responsible for:

- creating the split itself
- batching or shuffling
- model-specific logic
- augmentation policy
- caching features to disk
- training loop behavior

Keeping the dataset loader focused in this way helps prevent the data pipeline from becoming overly coupled or difficult to extend.

### Using the saved split manifest as the source of truth

A key design choice was to make the saved split CSV the authoritative record of dataset membership. Rather than rediscovering files from disk every time training begins, the dataset loader reads the existing manifest and filters it to the selected split.

This has two main benefits.

First, it guarantees consistency. If the same split file is used across multiple runs, then the exact same samples belong to training, validation, and test each time.

Second, it keeps the responsibilities clean. The indexing and splitting modules determine **which** files belong to each subset, while the dataset loader determines **how** to load those files into memory.

### Lazy loading of audio files

The dataset loader does not preload all audio files into memory when it is initialized. Instead, it stores the filtered manifest rows and loads the actual WAV file only when a specific sample is requested.

This is commonly referred to as **lazy loading**.

For example, when the `train` dataset object is created, it stores the list of training rows from the split manifest, including information such as file path, label, machine type, machine ID, dB condition, and group ID. However, the waveform data itself is not loaded until `__getitem__(i)` is called for a particular sample index.

This design was chosen for practical reasons:

- it keeps memory usage manageable
- it keeps initialization lightweight
- it allows later preprocessing behavior to remain flexible
- it aligns naturally with PyTorch’s dataset and dataloader workflow

### Sample retrieval through `__getitem__`

The most important function in the dataset class is `__getitem__(i)`. This function defines what happens when the pipeline asks for one specific sample.

At a high level, the sequence is:

1. locate the `i`-th row in the filtered split manifest
2. read the WAV file at the stored file path
3. obtain the raw waveform and sample rate
4. convert the text label into a numeric binary target
5. package the sample into a consistent output structure
6. optionally pass that sample through a later transform hook

This means the dataset loader produces one usable sample at a time. The PyTorch `DataLoader` can then call `__getitem__` repeatedly to construct batches.

### Label handling

The project frames anomaly detection as a binary classification problem. For the baseline dataset loader, the labels are encoded as:

- `normal -> 0`
- `abnormal -> 1`

This conversion happens directly in the dataset layer so that later training code receives numeric targets rather than raw label strings.

This choice keeps the target format simple and explicit, while still leaving room for later loss-specific type conversion if needed.

### Returned sample structure

Rather than returning a simple tuple immediately, the dataset loader returns each sample as a dictionary. In the baseline version, the returned structure contains:

- `waveform`
- `sample_rate`
- `label`

and optionally:

- `metadata`

This decision was made to keep the interface flexible. At this stage of the project, the final model input representation has not yet been fixed, because deterministic preprocessing such as spectrogram or log-mel conversion is the next task. Returning a dictionary makes it easier to preserve both the raw waveform and the supporting information needed for the next stage.

The metadata field is useful for bookkeeping, debugging, and later analysis. It includes values such as:

- file path
- machine type
- machine ID
- dB condition
- group ID
- split name

These fields are not intended as model inputs for the baseline. Instead, they help trace each sample back to its source conditions when needed.

### Audio loading behavior

The dataset loader standardizes the raw loaded audio into a consistent in-memory format. This is an important part of the design, because downstream preprocessing code depends on receiving audio in a predictable structure.

The raw waveform is returned as a tensor with shape:

`[channels, num_samples]`

and the sample rate is returned together with it.

The sample rate was checked during verification and was found to be consistent at `16000` Hz for the tested files. This matches the expected dataset format and is important because later audio preprocessing operations such as spectrogram or log-mel conversion depend on the sample rate being known correctly.

### Verified multichannel raw audio

An important finding during dataset verification was that the loaded waveform shape was:

`[8, 160000]`

for the tested samples across train, validation, and test splits.

This means the raw WAV files in the current setup are **multichannel** rather than mono. In other words, each sample contains 8 synchronized audio channels, each with 160000 samples. At a sample rate of 16000 Hz, this corresponds to approximately 10 seconds of audio per channel.

This is not a problem in the dataset loader itself. On the contrary, it shows that the loader is correctly exposing a real property of the data. However, it does affect the next design stage. Specifically, preprocessing will need to decide how the baseline should handle multichannel audio, for example by:

- preserving all channels
- reducing them to a single channel
- extracting features per channel and combining them later

This question belongs to preprocessing rather than data loading, but the dataset verification step was important because it revealed that this decision must be made explicitly.

### Manifest validation and defensive checks

The dataset loader also performs light validation when reading the manifest and accessing files. This includes checking that:

- the split manifest exists
- required columns are present
- the requested split is valid
- the selected split is not empty
- labels are valid
- referenced audio file paths exist

These checks are useful because they cause failures to appear early and clearly, rather than allowing problems to surface later during training in a less interpretable way.

### Relationship to preprocessing and augmentation

The dataset loader was intentionally implemented before deterministic preprocessing. At this stage, it is responsible only for reliable raw WAV loading and target preparation.

The next step in the pipeline is to add deterministic preprocessing, such as waveform-to-spectrogram or waveform-to-log-mel conversion. This should be integrated through a clean transform hook rather than being deeply hardcoded into the dataset itself.

Train-time augmentation is a later concern and is conceptually separate from preprocessing. Preprocessing defines the standard representation the model will use, while augmentation introduces additional variation during training to improve robustness. For this reason, augmentation was intentionally left out of the first dataset loader version.

### Role of the dataset loader in the full pipeline

The dataset loader can be viewed as the stage that turns the saved split artifact into actual sample objects that PyTorch can work with.

The broader pipeline now looks like:

1. raw WAV files are stored under `data/raw/`
2. the indexing and splitting pipeline creates a saved split manifest
3. `dataset.py` reads that manifest and exposes one split as a PyTorch dataset
4. each call to `__getitem__` loads one raw sample and target label
5. deterministic preprocessing will next convert that raw waveform into a model-ready representation
6. the PyTorch `DataLoader` will batch those processed samples for training or evaluation

This separation helps keep the pipeline organized and makes each stage easier to test independently.

### Summary

In summary, the dataset loader was implemented as a lightweight but important bridge between the saved split manifest and the later preprocessing and training stages. It reads a fixed split file, filters one selected subset, lazily loads raw WAV files, encodes binary labels, and returns a consistent sample structure. The verification step confirmed that the loader works across train, validation, and test splits, and also revealed that the current raw audio is multichannel with waveform shape `[8, 160000]` at `16000` Hz. This finding does not change the purpose of the dataset loader, but it does directly inform the next preprocessing design decision.

---

## Baseline Preprocessing Pipeline

After verifying that the dataset loader could reliably read the saved split manifest and load raw WAV files, the next step was to define the baseline preprocessing pipeline. The purpose of preprocessing in this project is to convert the raw audio waveform into a more structured and model-ready representation while keeping the baseline simple, deterministic, and reproducible.

At this stage, the goal was not to explore every possible audio representation. Instead, the aim was to choose one clean and practical baseline preprocessing path that could later serve as a stable reference point for future experiments.

### Why preprocessing is needed

Although the dataset loader can already return raw waveform tensors, most baseline audio classification models do not work directly on the raw waveform without additional design considerations. In this project, the baseline model is intended to be a CNN-based classifier, and CNNs are often easier to apply to structured time-frequency representations such as spectrograms or log-mel spectrograms.

Preprocessing therefore serves two main purposes:

1. convert the raw waveform into a representation that is easier for the baseline model to consume
2. standardize the input format so that all training, validation, and test samples are processed consistently

This stage was intentionally kept deterministic. No augmentation was included here, because augmentation is conceptually separate from defining the standard baseline input representation.

### Separation between raw loading and preprocessing

A deliberate design decision was made to keep raw audio loading separate from preprocessing.

The dataset loader in `src/data/dataset.py` remains responsible for:

- reading the saved split manifest
- filtering by split
- loading the raw WAV file
- returning waveform, sample rate, label, and optional metadata

The preprocessing stage was then implemented through a transform hook rather than being hardcoded directly into the dataset loader. This keeps responsibilities cleaner:

- the dataset loader handles raw data access
- waveform utilities handle channel-level waveform processing
- spectrogram utilities handle feature extraction
- the transform module combines these steps into one baseline preprocessing pipeline

This separation makes the pipeline easier to test, easier to understand, and easier to modify later if different preprocessing methods need to be compared.

### Verified raw audio properties

Before finalizing preprocessing, the raw audio output from the dataset loader was inspected. A key finding was that the waveform shape was consistently:

`[8, 160000]`

with sample rate:

`16000 Hz`

This means that each audio sample is multichannel, containing 8 synchronized channels and 160000 samples per channel. At a sample rate of 16000 Hz, this corresponds to approximately 10 seconds of audio.

This verification step was important because it revealed that preprocessing needed to make an explicit decision about how to handle multichannel input.

### Why resampling was not needed

One common preprocessing step in audio pipelines is resampling all files to a shared target sample rate. In this project, however, resampling was not included in the baseline preprocessing pipeline.

The reason is that the sample rate had already been checked during dataset loading and verification, and the tested samples were consistently found to use the expected `16000 Hz` rate. Because the current dataset already matches the intended target rate, there was no need to introduce an additional resampling step.

For the baseline, sample rate handling was therefore treated as a validation condition rather than an active transformation. If an unexpected sample rate appears later, it should raise an error instead of silently altering the waveform.

This decision keeps the baseline simpler and avoids introducing unnecessary processing when the data is already in the required format.

### Why input length adjustment was not needed

Another common preprocessing step in audio tasks is trimming, padding, or segmenting audio clips so that all inputs have the same length. In this project, that step was also not included in the first baseline preprocessing version.

The reason is that the verified raw waveform shape was already consistent across the tested samples. Each loaded clip had length `160000` samples per channel, which corresponds to a fixed clip duration at `16000 Hz`.

Because the current data already provides a consistent input length, there was no immediate need to trim, pad, or segment the audio further. Adding such logic at this stage would have increased complexity without addressing a current problem.

This does not mean variable-length handling is never useful. Rather, it means that for the present baseline, the fixed-length property of the data allows this step to be safely omitted.

### Deciding how to handle 8-channel audio

Once it was established that the raw waveform was multichannel, the next preprocessing decision was how the baseline should use those 8 channels.

Several possibilities were considered, such as:

- keeping all 8 channels
- selecting one channel only
- extracting features for each channel separately
- reducing the multichannel waveform to a single mono waveform

For the baseline, the chosen approach was to **average the 8 channels to mono**.

### Why channel averaging was chosen

Averaging the channels to produce a mono waveform was chosen for practical baseline reasons.

First, it keeps the first preprocessing pipeline simple. Instead of immediately designing a multichannel feature pipeline and a model that explicitly handles 8 input channels, the baseline can focus on establishing one standard and stable representation.

Second, it reduces the dimensionality of the input while still using information from all channels. Unlike selecting a single channel, averaging does not arbitrarily discard seven channels entirely. Instead, it combines them into one waveform in a straightforward and deterministic way.

Third, this choice makes the later baseline model design easier. A mono waveform can be converted into a single log-mel spectrogram, which then naturally fits a conventional CNN input structure.

This does not imply that using all 8 channels is always worse. It only means that for the baseline, a mono representation was preferred in order to prioritize clarity, stability, and ease of interpretation.

### Waveform-to-feature conversion

After reducing the raw waveform to mono, the next step was to convert it into a time-frequency representation. The chosen baseline representation was a **log-mel spectrogram**.

This was selected because it is a common and practical input representation for audio classification tasks. Compared with a raw waveform, a log-mel spectrogram makes the frequency content of the signal more explicit over time, which is often more suitable for CNN-based baseline models.

The feature extraction process follows a sequence of signal-processing steps:

1. start from the mono waveform
2. apply the Short-Time Fourier Transform (STFT)
3. compute a power spectrogram
4. apply a mel filterbank
5. apply a log-scale transform
6. return the resulting log-mel spectrogram

### Short-Time Fourier Transform (STFT)

The first step in feature extraction is the Short-Time Fourier Transform.

A raw waveform describes how the signal amplitude changes over time, but it does not directly show which frequencies are present at different moments. The STFT addresses this by dividing the waveform into short overlapping windows and computing a frequency decomposition for each window.

This produces a time-frequency representation: instead of seeing only one long waveform, the pipeline now obtains information about how spectral content evolves over time.

For the baseline, fixed STFT parameters were chosen so that preprocessing remains deterministic and reproducible. These settings were not treated as heavily optimized yet; they serve as stable baseline defaults.

### Power spectrogram

The direct output of the STFT contains complex-valued frequency coefficients. For baseline feature extraction, the interest is not in the complex phase values themselves but in the energy distribution across frequencies.

For this reason, the STFT output is converted into a **power spectrogram**. This represents the signal energy at each time-frequency location.

Using a power spectrogram provides a more interpretable and commonly used intermediate representation for later mel scaling and logarithmic compression.

### Applying the mel filterbank

The next step is to project the power spectrogram onto the **mel scale**.

The mel scale groups frequencies into perceptually motivated bands rather than keeping every raw frequency bin independently. In practice, this reduces the dimensionality of the frequency axis while preserving broad spectral structure.

For the baseline, this produces a **mel spectrogram**, which is more compact and easier to use as input for the model than the full raw STFT representation.

Even though the project concerns machine sounds rather than speech, using mel-scaled features still provides a practical and widely used baseline representation for time-frequency learning.

### Log transform and final log-mel spectrogram

After the mel spectrogram is computed, a logarithmic transform is applied. This produces the final **log-mel spectrogram**.

The purpose of the log transform is to compress the dynamic range of the feature values. In linear scale, high-energy regions can dominate numerically while low-energy regions become difficult to represent clearly. The log transform reduces this imbalance and makes the representation more suitable for learning.

This final log-mel output is therefore easier for the baseline CNN to consume than either the raw waveform or an uncompressed linear-scale spectrogram.

### Final baseline preprocessing output

The preprocessing transform returns a dictionary containing:

- `input`
- `label`
- `metadata` (when metadata is enabled)

The raw waveform and sample rate are used internally during preprocessing but are not kept in the final transformed sample for baseline model input.

The final processed tensor is arranged so that it is ready for a CNN-style input pipeline, with a single feature channel and the log-mel representation as the main model input.

### Why no additional amplitude normalization was added

Another possible preprocessing decision was whether to explicitly normalize each waveform amplitude to a fixed range such as `[-1, 1]`.

For the first baseline, no extra per-clip amplitude normalization step was added. The waveform values loaded by the audio library were used directly, and preprocessing focused on channel averaging and log-mel feature extraction.

This choice was made to avoid introducing an additional transformation that changes the relative amplitude structure of each clip before there is evidence that such normalization is necessary. Since the current goal is to establish a simple and stable baseline, it was preferable to avoid adding extra signal manipulation unless it solves a clear problem.

### On-the-fly preprocessing

The preprocessing pipeline was implemented to run **on the fly** through the dataset transform mechanism.

This means that raw WAV files are still stored under `data/raw/`, and the deterministic preprocessing steps are applied when a sample is requested from the dataset, rather than being precomputed and saved to disk immediately.

This approach was chosen because:

- it keeps the baseline implementation simple
- it avoids committing too early to a cached feature format
- it remains flexible if preprocessing settings need to be adjusted later
- it fits naturally into the PyTorch dataset and dataloader workflow

Feature caching may still be useful later, but it was intentionally deferred until the baseline preprocessing design became stable.

### Verification of the preprocessing pipeline

After implementation, the preprocessing pipeline was checked using a small sanity-check script. This confirmed that a sample could be loaded through the dataset with the preprocessing transform attached, and that the transformed output had the expected structure:

- `input`
- `label`
- `metadata`

The processed input shape was also verified to be consistent with the baseline design.

This verification step was useful because it confirmed that preprocessing was correctly integrated into the dataset pipeline before moving on to the baseline model and training stages.

### Summary

In summary, the baseline preprocessing pipeline was designed to stay simple, deterministic, and closely aligned with the verified properties of the dataset. Because the audio was already consistently sampled at `16000` Hz and had a fixed input length, neither resampling nor length adjustment was needed in the first baseline version. The main preprocessing decision was therefore how to handle the 8-channel raw waveform. For the baseline, the channels were averaged to mono in order to simplify the feature pipeline while still incorporating information from all channels.

The mono waveform was then converted into a log-mel spectrogram by applying the STFT, forming a power spectrogram, projecting it through a mel filterbank, and finally applying a logarithmic transform. This produced a compact and model-friendly time-frequency representation that serves as the standard baseline input for the next stage of the project.

---

## Baseline Model and Training Pipeline

After defining the saved split strategy, implementing the dataset loader, and verifying the baseline preprocessing pipeline, the next stage of the project was to build the first trainable baseline model and connect it to a full training loop.

At this point, the objective was not to build the strongest possible system immediately. Instead, the goal was to establish a **simple, stable, and reproducible baseline training pipeline** that could later serve as a reference point for future experiments. This meant choosing a model that was easy to implement, compatible with the current preprocessing output, and straightforward to train and debug.

### Why a CNN was chosen for the baseline

The baseline preprocessing pipeline converts each audio sample into a **log-mel spectrogram**. This representation is naturally structured as a 2D time-frequency map, where one axis corresponds to frequency bins and the other corresponds to time frames. Because of this, a Convolutional Neural Network (CNN) is a practical first choice.

A CNN was chosen for several reasons.

First, CNNs are well suited to learning local patterns in structured 2D inputs. In a log-mel spectrogram, meaningful information often appears as local time-frequency structures rather than as isolated individual values. Convolutional filters can learn these local patterns more naturally than a fully connected model.

Second, CNNs are a common and reliable baseline for audio classification tasks using spectrogram-like inputs. Since one of the main goals of this project is to build a stable experimental foundation, using a standard and interpretable model architecture makes the results easier to understand and compare later.

Third, a CNN provides a good balance between simplicity and representational power. It is more structured than flattening the spectrogram into a single vector and feeding it into a multilayer perceptron, but it is still much simpler than immediately moving to larger or pretrained models. This matches the project principle of starting with a practical baseline before introducing more complex methods.

### Why more complex models were not used first

Although stronger architectures may ultimately perform better, they were intentionally deferred at this stage.

The project priorities at this point were:

- establish a working end-to-end pipeline
- confirm that the split, loading, preprocessing, model, and training stages work together correctly
- create a baseline result that can later be improved in controlled experiments

Jumping directly to pretrained or more complex architectures would have made it harder to determine whether later performance gains came from the model itself or from unresolved issues elsewhere in the pipeline. For this reason, the first model was kept deliberately modest.

### Baseline model input and output contract

The baseline model was designed to match the output of the preprocessing pipeline directly.

After deterministic preprocessing, each sample is represented as a tensor with shape:

`[1, 64, 313]`

where:

- `1` is the feature channel dimension
- `64` is the number of mel bins
- `313` is the number of time frames

When batched by a PyTorch `DataLoader`, the model therefore receives input with shape:

`[batch_size, 1, 64, 313]`

This design keeps the interface between preprocessing and the model clear and consistent.

The task is binary classification with labels:

- `normal -> 0`
- `abnormal -> 1`

For this reason, the model was designed to output **one logit per sample** rather than two class scores. This means the output shape is:

`[batch_size]`

Using one logit is a natural fit for binary classification and keeps the final classifier head simpler.

### Why one logit was used instead of two class scores

A binary classification model can be implemented in two common ways:

1. output two class scores and use a multiclass loss such as cross-entropy
2. output one logit and use a binary loss such as binary cross-entropy with logits

For the baseline, the second approach was chosen.

This choice was made because the problem is directly binary, the dataset labels are already encoded as `0` and `1`, and a one-logit output keeps the model head and loss pairing straightforward. It also avoids adding an unnecessary second output unit when only one binary decision is required.

As a result, the training loss for the baseline was set to:

`BCEWithLogitsLoss`

This combines the sigmoid operation and binary cross-entropy into one numerically stable loss function, which is preferable to applying a sigmoid manually inside the model during training.

### Baseline CNN architecture design

The baseline CNN was intentionally kept small and conventional.

Its overall structure follows the pattern:

1. convolutional layers to learn local time-frequency features
2. nonlinear activation after each convolution
3. batch normalization for more stable training
4. dropout for light regularization
5. max pooling to reduce feature map size
6. global pooling and fully connected layers for final classification

The model uses three convolutional blocks rather than a very shallow one- or two-layer design. This was chosen as a practical middle ground: the model is still simple, but it has enough depth to learn progressively more abstract features from the log-mel input.

The final classifier head outputs one scalar logit for each sample.

### Why adaptive pooling was used

An important design choice in the baseline CNN was the use of **adaptive average pooling** before the fully connected classifier.

A simpler but more fragile alternative would have been to flatten the final convolutional feature map using a hardcoded size. However, that approach ties the model more tightly to one exact intermediate tensor shape and makes the model more brittle if input dimensions change slightly later.

Adaptive pooling was therefore chosen because it:

- reduces dependence on a hardcoded flatten dimension
- keeps the classifier head simpler
- makes the model slightly more robust to changes in the time dimension

This choice improves maintainability without making the baseline architecture significantly more complicated.

### Keeping the model module focused

Another design decision was to keep the model implementation narrowly scoped.

The model module is responsible only for defining:

- network layers
- forward pass behavior
- expected input/output tensor structure

It is intentionally not responsible for:

- loss selection
- optimizer setup
- training loop logic
- checkpointing
- metric computation

These responsibilities belong to the training pipeline instead. Separating them makes the code easier to understand and allows the model to be reused later with different training settings if needed.

## Baseline Training Pipeline

Once the baseline CNN was defined, the next step was to connect it to a minimal but complete training pipeline.

The goal of this stage was to create an end-to-end training process that could:

- load training and validation data from the saved split manifest
- preprocess each sample on the fly
- train the baseline model using mini-batches
- monitor validation performance over time
- save useful checkpoints
- record training history for later analysis

As with earlier stages, the design emphasized modularity and reproducibility over premature optimization.

### Why mini-batch training was used

The model was trained using **mini-batch gradient descent** through a PyTorch `DataLoader`.

Mini-batch training was chosen instead of processing one sample at a time because it is the standard and practical approach for neural network training. It improves computational efficiency, especially on accelerators such as GPUs, and provides a more stable gradient estimate than purely single-sample updates.

The `DataLoader` also centralizes responsibilities such as:

- batching samples
- shuffling the training split
- iterating through the dataset cleanly across epochs

This makes the training code more organized and avoids manually handling low-level data iteration.

### Separation between training orchestration and epoch logic

The training pipeline was organized so that responsibilities remained clearly separated.

At a high level, the training code was split into:

- **epoch-level routines**, which handle one pass through the data
- **higher-level orchestration**, which manages the full multi-epoch training run

This separation was useful because the repeated mechanics of one epoch, such as forward pass, loss computation, backward pass, and metric accumulation, are conceptually different from higher-level tasks such as:

- creating datasets and dataloaders
- selecting the device
- initializing model, optimizer, and loss
- saving checkpoints
- recording history
- handling resume behavior

Keeping these responsibilities separate made the training code easier to read, easier to test, and easier to extend later.

### Device selection strategy

The training pipeline was designed to run on the best available device in the following priority order:

1. CUDA
2. MPS
3. CPU

This allows the same training code to run on different hardware environments without changing the core logic. CUDA support is preferred when an NVIDIA GPU is available, MPS is used as a fallback on Apple Silicon systems that support it, and CPU remains the final default.

This design improves portability while preserving a single consistent training workflow.

### Why BCEWithLogitsLoss was used

Because the model outputs one logit per sample for binary classification, the training loss was chosen to be:

`BCEWithLogitsLoss`

This was preferred over applying a sigmoid manually inside the model because it is the standard numerically stable implementation for binary logistic output training.

This choice also reinforces a clean separation of roles:

- the model outputs raw logits
- the loss function handles the appropriate binary classification objective
- sigmoid is only used outside the model when probabilities or thresholded predictions are needed for metrics

### Why Adam was chosen as the optimizer

For the first baseline, the optimizer was chosen to be **Adam**.

Adam was selected because it is a practical and reliable default for baseline deep learning experiments. It typically converges more easily than plain stochastic gradient descent in early-stage projects and usually requires less manual tuning to produce a working baseline.

At this stage, the main objective was not to optimize the training procedure exhaustively, but to get a stable baseline running with a widely accepted optimizer choice. Adam fits that requirement well.

### Training and validation each epoch

The training process was organized around epochs. For each epoch, the pipeline performs two separate phases:

1. **training phase**
2. **validation phase**

During the training phase, the model is placed in training mode, mini-batches are loaded from the training set, logits are computed, loss is evaluated, gradients are backpropagated, and the optimizer updates the parameters.

During the validation phase, the model is placed in evaluation mode, gradients are disabled, and the model is evaluated on the validation set without updating parameters.

This separation is important because the validation set is used to monitor generalization performance rather than optimize the model directly.

### Why validation was used every epoch

Validation was run at the end of every epoch rather than every few epochs.

This choice was made because validation provides direct feedback on how the model is performing on held-out data as training progresses. If validation is performed too infrequently, the best-performing model state may be missed.

Running validation every epoch makes it possible to:

- track validation loss over time
- compare training and validation behavior for signs of overfitting
- save the best checkpoint at the exact epoch where validation performance is strongest

For a baseline training loop, this is a practical and standard schedule.

### Monitoring overfitting

One reason for including validation every epoch was to monitor possible overfitting.

Overfitting is not identified merely by a low training loss. Instead, it becomes visible when training performance continues to improve while validation performance stops improving or begins to worsen.

For this reason, the training pipeline records at least:

- training loss
- validation loss
- validation accuracy

This allows the baseline run to be interpreted more meaningfully than if only training loss were observed.

### Why validation loss was used for checkpoint selection

The main checkpoint selection criterion for the baseline was **best validation loss**.

This was chosen over more complex checkpoint rules because validation loss is:

- directly tied to the optimization objective
- smooth and continuous
- less dependent on an explicit probability threshold than classification accuracy

For a first baseline, this makes validation loss a cleaner and more stable checkpoint criterion than metrics such as accuracy, F1 score, or ROC-AUC.

This does not mean those metrics are unimportant. Rather, it means that validation loss was treated as the primary checkpoint decision signal in the initial training pipeline.

### Two checkpoint purposes: best and last

Checkpointing was included in the baseline pipeline for two different reasons.

#### Best checkpoint

The first purpose is to preserve the best-performing model according to validation loss.

Whenever validation loss improves, the current model state is saved as the **best checkpoint**. This checkpoint is intended for later evaluation on the test set and for reporting the baseline result.

#### Last checkpoint

The second purpose is recovery and resume capability.

At the end of every epoch, the current training state is saved as the **last checkpoint**. This ensures that if training is interrupted unexpectedly, it can be resumed without restarting from scratch.

This distinction is important because the best checkpoint and the most recent checkpoint are not necessarily the same. A model can continue training after its best validation epoch, which means the latest state may not be the best one.

### What was stored in checkpoints

To support meaningful checkpointing, the saved training state includes at least:

- model parameters
- optimizer state
- current epoch
- best validation loss so far

Including optimizer state is especially important for resume functionality, because it allows adaptive optimizers such as Adam to continue from their current internal state rather than restarting as if training had just begun.

### Training history and timing

In addition to checkpoints, the training pipeline records a lightweight history over time.

This history includes at least:

- epoch number
- training loss
- validation loss
- validation accuracy
- epoch duration
- total training time

Recording this history serves several purposes.

First, it makes later analysis easier, because the behavior of the model over time can be reviewed after training ends.

Second, it supports plotting learning curves such as training loss and validation loss across epochs.

Third, it improves experimental reproducibility by preserving more than just a final checkpoint. The run can be interpreted in terms of how it evolved, not only where it ended.

### Why loss curves are useful

The loss curve was treated as an important diagnostic artifact rather than a cosmetic extra.

A training-only loss curve can be misleading because it does not show how well the model generalizes. By recording both training loss and validation loss over time, it becomes easier to detect patterns such as:

- healthy joint improvement
- underfitting
- overfitting
- unstable optimization

For this reason, the training history was designed so that these curves can be plotted later even if plotting is not the first implementation priority.

### Reproducibility during training

Reproducibility remained an important principle during the training stage as well.

To support more consistent runs, the training pipeline includes seed control so that random number generation for model initialization and other stochastic components is more controlled. This does not guarantee bitwise-identical behavior across every environment, but it does improve consistency and aligns with the overall project goal of producing reproducible experiments.

### Why final test evaluation was separated from training

Although validation is part of the training pipeline, final test evaluation was intentionally treated as a separate later stage.

This separation was chosen because the test set should not guide training decisions. The validation set exists for monitoring and checkpoint selection during model development, while the test set is intended for the final held-out assessment of the chosen model.

Keeping the final evaluation step separate helps preserve the integrity of the test set and makes the experimental procedure easier to explain.

### Summary

In summary, the baseline training stage was designed to transform the previously prepared data pipeline into a complete learning workflow. A CNN was chosen because it is a practical and well-matched baseline model for 2D log-mel spectrogram inputs. The model was intentionally kept simple, with one-logit binary output and a standard convolutional architecture, in order to prioritize clarity, debuggability, and reproducibility.

The training pipeline then connected this model to mini-batch dataloading, device-aware training, `BCEWithLogitsLoss`, the Adam optimizer, epoch-level training and validation, validation-based checkpointing, resume support, and saved training history. Together, these decisions created a baseline training framework that is sufficiently complete to support initial experiments while remaining modular enough for future extensions.

---

## Baseline Evaluation Pipeline

After confirming that the baseline training pipeline was working correctly, the next step was to implement a separate evaluation stage. The purpose of this stage was to assess how well the trained model generalizes to **held-out test data** that was not used for gradient updates or checkpoint selection.

At this point in the project, the main goal was not to build a large evaluation framework. Instead, the aim was to create a **simple, stable, and reproducible test-time evaluation pipeline** that fits cleanly into the existing modular structure.

### Why evaluation was kept separate from training

Although validation is already performed during training, final evaluation was intentionally implemented as a separate step.

This distinction is important because validation and test evaluation serve different purposes:

- **validation** is used during training to monitor learning progress and select checkpoints
- **test evaluation** is used after training to estimate generalization on fully held-out data

If the test set is used repeatedly during model development, it stops serving as a clean final benchmark. For this reason, the project keeps the test split separate and uses it only after training has already selected a model.

This design also keeps responsibilities cleaner:

- the training pipeline handles optimization and validation
- the evaluation pipeline handles final test inference and metric computation

### Why the best checkpoint was used instead of the last checkpoint

A key design decision in the evaluation stage was to evaluate the **best checkpoint**, not the last checkpoint.

During training, two checkpoint types are saved:

- **best checkpoint**, saved whenever validation loss improves
- **last checkpoint**, saved every epoch for resume/recovery

These two checkpoints serve different purposes. The last checkpoint is useful if training is interrupted and needs to continue later, but it is not necessarily the strongest-performing model. A later epoch may have a lower training loss while already beginning to overfit relative to validation performance.

For this reason, the final evaluation stage uses the checkpoint selected by **lowest validation loss**. This keeps the evaluation aligned with the model selection rule already defined during training.

### Reusing the same baseline data pipeline

The evaluation stage was intentionally built on top of the same dataset and preprocessing pipeline already used for training.

The evaluation script:

1. loads the saved split manifest
2. creates the dataset for the `test` split
3. applies the same deterministic baseline preprocessing transform
4. loads the saved best checkpoint
5. runs inference across the test set without updating model parameters

This was an important design choice because it ensures consistency between training and evaluation. The only difference is the selected data split and the fact that gradients are disabled.

By reusing the same dataset loader and preprocessing transform, the project avoids creating a separate test-only preprocessing path that could silently drift away from the training setup.

### Test-time inference behavior

During evaluation, the model is placed in evaluation mode and inference is run under `torch.no_grad()`.

This serves two purposes:

- it ensures layers such as dropout behave appropriately for inference
- it avoids storing gradients unnecessarily, which reduces memory use and makes evaluation more efficient

The model outputs one logit per sample, consistent with the binary classification design used during training.

### Converting logits into predictions

Because the model outputs a single logit for each sample, an explicit conversion step is needed during evaluation.

The process is:

1. apply the sigmoid function to convert the logit into a probability
2. threshold the probability at `0.5`
3. assign:
   - `0` for `normal`
   - `1` for `abnormal`

This decision rule was kept intentionally simple for the baseline. No threshold tuning was performed at this stage, because the first goal was to establish a default and reproducible evaluation procedure rather than optimize operating thresholds.

### Why these baseline metrics were chosen

The first baseline evaluation was designed to compute a compact but useful set of binary classification metrics. The chosen metrics were:

- test loss
- accuracy
- precision
- recall
- F1 score
- confusion matrix
- ROC-AUC

These metrics were chosen because each one provides a slightly different view of model behavior.

#### Test loss

Test loss provides a threshold-independent view of how well the model’s outputs align with the target labels under the same loss function used during training. It is useful as a direct continuation of the training objective.

#### Accuracy

Accuracy gives a simple overall summary of how many samples were classified correctly. However, accuracy alone can be misleading if one class is easier to predict than the other, so it was not treated as the only metric of interest.

#### Precision and recall

Precision and recall were included because anomaly detection is often not well described by accuracy alone.

- **precision** answers: when the model predicts `abnormal`, how often is it correct?
- **recall** answers: out of all truly abnormal samples, how many did the model successfully detect?

This distinction is especially important because a model can appear strong overall while still missing a meaningful portion of abnormal samples.

#### F1 score

The F1 score combines precision and recall into a single value and is useful when both false positives and false negatives matter.

#### Confusion matrix

The confusion matrix was included because it gives the raw count-level breakdown of:

- true negatives
- false positives
- false negatives
- true positives

This makes it easier to interpret precision and recall in concrete terms rather than only as summary statistics.

#### ROC-AUC

ROC-AUC was also included in the baseline because it evaluates how well the model separates the two classes across thresholds rather than only at one threshold such as `0.5`.

This is helpful because a model may have strong ranking ability even if a fixed threshold is not yet optimal. Including ROC-AUC therefore provides additional perspective on class separability.

### Why evaluation results were saved as artifacts

The evaluation stage does not only print results to the console. It also saves the computed metrics to disk as a structured artifact.

This decision supports reproducibility and comparison. If evaluation results existed only in terminal output, they would be easy to lose and difficult to compare later across runs or machine types.

Saving the test metrics to a JSON artifact makes it possible to:

- reproduce reported results more easily
- compare multiple runs consistently
- generate later plots or summary tables
- keep the project outputs aligned with the broader artifact-based workflow

### Folder-based output organization

As the pipeline became more complete, the artifact organization was also improved.

Rather than saving all checkpoints and metric files into flat directories, outputs were grouped into **run-specific folders**. This was chosen because even a small project can quickly accumulate many artifacts once multiple runs or machine types are evaluated.

For example, organizing outputs by run makes it easier to keep together:

- checkpoints
- training history
- run configuration
- test metrics
- generated plots

This improves clarity and makes later comparison across machine types easier.

### Baseline evaluation result on the first machine-type run

After implementing the evaluation pipeline, the first baseline test run was performed on the saved `fan` split using the best checkpoint.

The observed test results were:

- test loss: `0.1001`
- accuracy: `0.9603`
- precision: `0.9763`
- recall: `0.8742`
- F1 score: `0.9224`
- ROC-AUC: `0.9920`

These results suggest that the baseline model generalizes well on the held-out fan test split. Accuracy and F1 score were strong, and ROC-AUC was especially high, indicating that the model separates normal and abnormal samples effectively across thresholds.

One notable pattern is that precision was higher than recall. This means the model was very accurate when it predicted `abnormal`, but it still missed some abnormal samples. In other words, the baseline appears somewhat conservative in predicting anomalies. This is not necessarily a flaw, but it is an important behavior to note because anomaly detection tasks often care about missed abnormal cases.

### Why the first evaluation stage was kept intentionally simple

Even though more analysis could have been added, the first evaluation stage was intentionally kept small and practical.

For example, the baseline evaluation did **not** yet include:

- threshold tuning
- calibration analysis
- per-group error breakdown
- extensive error analysis reports

These were deferred because the immediate priority was to confirm that the baseline model could be trained, selected, and evaluated end to end in a reproducible way. Once this baseline is stable, more detailed analysis can be added in later experiments if needed.

### Summary

In summary, the evaluation stage was designed as a clean final pass over the held-out test split using the saved best checkpoint. It reuses the same dataset and preprocessing pipeline as training, converts one-logit model outputs into binary predictions through a sigmoid plus threshold rule, and computes a compact but informative set of binary classification metrics. The results are saved as structured artifacts to support reproducibility and later comparison.

This stage completes the first end-to-end baseline by providing a reliable estimate of test-set performance without redesigning the earlier data, preprocessing, model, or training stages.

---

## Baseline Visualization and Reporting Artifacts

Once training and evaluation were both working, the next step was to add a small visualization stage. The purpose of this stage was not to create a large reporting framework, but rather to produce a few high-value plots that make the baseline behavior easier to interpret.

### Why visualization was added

The baseline pipeline already saves numerical artifacts such as training history and test metrics. However, some aspects of model behavior are easier to understand visually than through raw numbers alone.

For the baseline, visualization was added mainly to support three practical goals:

1. inspect training stability over time
2. check for signs of overfitting or unstable learning
3. interpret final test performance more clearly

This stage was treated as a lightweight reporting utility rather than a core part of model training.

### Why plotting was kept separate from training and evaluation

A deliberate design choice was to keep plotting separate from both the training pipeline and the evaluation pipeline.

This means that:

- training is still responsible for optimization and saving history
- evaluation is still responsible for test inference and metric computation
- plotting is only responsible for turning saved artifacts into visual summaries

This separation was chosen because plots are useful for interpretation, but they are not required for the correctness of training or evaluation. Keeping them separate makes the system easier to maintain and avoids coupling numerical computation with reporting code.

### Chosen baseline plots

For the first baseline, only three plots were prioritized:

- training loss vs validation loss over epochs
- validation accuracy over epochs
- test confusion matrix

These were chosen because they provide the most useful insight for relatively little additional complexity.

### Training loss vs validation loss

The training and validation loss curves were included because they are one of the clearest ways to inspect how learning progressed over time.

These curves help answer questions such as:

- did both training and validation improve together?
- did validation stop improving while training continued to improve?
- did optimization appear stable or noisy?

This makes them especially useful for diagnosing underfitting, overfitting, or unstable training behavior.

### Validation accuracy curve

Validation accuracy was also plotted as a complementary view of held-out performance across epochs.

Although validation loss remained the main checkpoint selection criterion, plotting validation accuracy still provides an intuitive view of how classification performance evolved over time.

### Test confusion matrix

The confusion matrix was chosen as the main evaluation visualization because it makes the final prediction behavior concrete.

Unlike summary metrics alone, the confusion matrix directly shows the count of:

- correctly identified normal samples
- normal samples incorrectly predicted as abnormal
- abnormal samples missed by the model
- correctly identified abnormal samples

This is particularly useful when precision and recall differ, because it helps explain where the trade-off is coming from.

### Why more advanced plots were not added yet

More complex visualizations were intentionally deferred. For example, the baseline visualization stage does not yet focus on:

- ROC curve plots
- calibration curves
- per-machine or per-dB breakdown charts
- embedding visualizations
- per-sample spectrogram inspection dashboards

These may still be useful later, but they were not necessary to establish the first baseline. The current priority was to produce a small number of plots that directly support interpretation of the baseline results.

### Saving plots as artifacts

Just like checkpoints and metric files, plots were also saved as run-specific artifacts rather than only displayed interactively.

This supports reproducibility, easier comparison between runs, and easier inclusion of visual outputs in later reporting. It also keeps the project aligned with the general principle that important outputs should be preserved as artifacts rather than left only in notebook cells or terminal sessions.

### Role of visualization in the full baseline pipeline

At this point, the baseline pipeline can be viewed as:

1. generate and save a split manifest
2. load raw WAV files through the dataset loader
3. apply deterministic preprocessing
4. train the baseline CNN
5. evaluate the saved best checkpoint on the test split
6. generate plots from saved history and metrics

This means the baseline is no longer only a model training script. It is now an end-to-end experimental workflow with saved split artifacts, saved checkpoints, saved metrics, and saved visualization outputs.

### Summary

In summary, a lightweight visualization stage was added after training and evaluation in order to make the baseline results easier to interpret. The selected plots were deliberately limited to loss curves, validation accuracy, and the confusion matrix, since these provide the most useful diagnostic and reporting value at this stage. Plotting was kept separate from the core training and evaluation logic so that the overall pipeline remains modular, reproducible, and easy to debug.

---

## Current Baseline Status and Next Step

At this point, the project has a complete baseline workflow for a single machine-type experiment:

- saved split generation
- raw WAV dataset loading
- deterministic baseline preprocessing
- baseline CNN model
- multi-epoch training with validation
- checkpointing and saved history
- held-out test evaluation
- baseline visualization

This means the project now has a stable reference pipeline that can be rerun and compared against future experiments.

The next logical step is not to redesign the baseline itself, but to apply the same baseline pipeline to the remaining machine types so that performance can be compared across:

- fan
- pump
- slider
- valve

This is an important stage because it helps determine whether the chosen baseline design works consistently across the broader dataset or whether performance differs substantially by machine type.

Only after this baseline comparison is complete does it make sense to move on to larger experimental changes such as augmentation, stronger architectures, threshold tuning, or multi-machine training.

---

## Multi-Machine Baseline Run and Result Interpretation

After completing the initial single-machine baseline, the same baseline pipeline was extended to an **all-machine** experiment covering:

- fan
- pump
- slider
- valve

The purpose of this run was not to redesign the model, preprocessing, or training setup. Instead, the aim was to test whether the same baseline pipeline could still perform well when trained and evaluated on a broader scope of the dataset.

### Why the all-machine run was important

The earlier single-machine baseline was useful as a first reference point, but it only showed that the pipeline worked on one selected subset. A stronger baseline should also be tested on a wider problem scope.

The all-machine run was therefore important for two reasons:

1. it checks whether the baseline remains effective when the dataset includes multiple machine types rather than just one
2. it provides a stronger reference point for future experiments, since later methods can be compared against both single-machine and all-machine baseline performance

This step also helps distinguish between two possibilities:

- the baseline only works well on a narrow subset such as `fan`
- the baseline captures more general patterns that remain useful across the broader dataset

### Split design used for the all-machine baseline

Because the all-machine result appeared very strong, an important follow-up question was whether the result could be caused by accidental train/validation/test leakage.

To investigate this, the split generation logic for the all-machine scope was reviewed more carefully.

The all-machine split was not created by pooling all files together and performing one global file-level random split. Instead, the split logic first applies the group-based split **within each machine type separately**, and only then concatenates the machine-specific split results into one combined manifest.

This means that:

- `fan` groups are split within `fan`
- `pump` groups are split within `pump`
- `slider` groups are split within `slider`
- `valve` groups are split within `valve`

before the final all-machine manifest is assembled.

This is an important sanity check because it means the train/validation/test assignments are still controlled at the group level rather than being mixed arbitrarily across the full dataset.

For the all-machine scope, the effective group identity is equivalent to:

`group = (machine_type, machine_id, snr_db)`

This is because each machine type is split separately and the final combined manifest prefixes group IDs with the machine name. As a result, groups remain distinct even when machine IDs and SNR values are similar across machine types.

### Observed split summary for the all-machine baseline

The saved all-machine split summary was:

- total files: `54057`
- train files: `36038`
- validation files: `8903`
- test files: `9116`

Class counts per split were:

- train: `abnormal=6600`, `normal=29438`
- validation: `abnormal=1529`, `normal=7374`
- test: `abnormal=1771`, `normal=7345`

Class proportions per split were:

- train: `abnormal=0.183`, `normal=0.817`
- validation: `abnormal=0.172`, `normal=0.828`
- test: `abnormal=0.194`, `normal=0.806`

Number of groups per split were:

- train: `32`
- validation: `8`
- test: `8`

These counts are consistent with the intended group-based split design. Each machine type contributes its own set of group units, and the final combined all-machine split preserves those assignments.

### Why the result initially seemed suspicious

The all-machine baseline produced very strong evaluation metrics, which made it reasonable to question whether the result was too good to be true.

The main concerns were:

- possible train/validation/test leakage
- possible metric or evaluation bugs
- overfitting
- the possibility that the combined all-machine scope might still be easier than expected because machine sounds within each type remain highly structured

These concerns were worth taking seriously, because unusually strong results are only meaningful if the evaluation procedure is valid.

### All-machine baseline training behavior

The all-machine training history showed steady improvement over epochs rather than an obviously broken pattern.

The best validation loss was reached at epoch 9, followed by a slight worsening at epoch 10. This is a normal training pattern and is consistent with mild end-of-training overfitting rather than a catastrophic bug.

This matters because if the split or metric computation had been fundamentally wrong, one might expect much more extreme or suspicious behavior, such as unrealistically perfect validation results or almost immediate convergence to near-zero loss. That was not observed here.

### All-machine baseline test results

Using the saved best checkpoint, the all-machine baseline test metrics were:

- test loss: `0.0636`
- accuracy: `0.9763`
- precision: `0.9887`
- recall: `0.8882`
- F1 score: `0.9358`
- ROC-AUC: `0.9952`

The confusion matrix was:

- true negatives: `7327`
- false positives: `18`
- false negatives: `198`
- true positives: `1573`

These are very strong results. However, they are not perfectly clean to the point of being automatically suspicious. The model still makes mistakes, especially by missing some abnormal samples. This is an important detail because it suggests that the model is not simply memorizing or producing an unrealistically perfect score.

### Interpreting the all-machine result

Based on the split logic review, training history, and evaluation outputs, the all-machine result appears to be **strong but plausible** under the current split design.

Several observations support this interpretation.

First, the split logic itself appears sound. The all-machine manifest is constructed from machine-type-specific group splits rather than from a naive pooled random split, which reduces the chance of accidental leakage across train, validation, and test sets.

Second, the metric pattern is internally consistent. Precision is extremely high, while recall is somewhat lower. This is similar to the earlier single-machine result and suggests a model that is somewhat conservative in predicting abnormalities. This is a believable pattern rather than an obviously broken one.

Third, the confusion matrix is strong but not unrealistically perfect. The presence of both false positives and false negatives suggests that the model is performing very well without being trivial.

### Important limitation of the current split design

Although the all-machine result appears valid for the current setup, it is still important to interpret it in context.

The current split design is **leakage-aware**, but it is not the strictest possible evaluation setting. The grouping rule keeps together files from the same machine type, machine ID, and SNR condition, but it does not completely hide an entire machine ID across all conditions in the same way that a stricter leave-one-machine-ID-out style split would.

This means the current result should be interpreted as:

- a strong result for the chosen group-based split design

rather than as proof that the model will necessarily generalize to the harshest possible unseen-machine scenario.

This is not a flaw in the current baseline. It simply means that the baseline result should be described accurately according to the evaluation setting used.

### Conclusion from the all-machine baseline

The current evidence suggests that the baseline CNN is genuinely performing well on the task **as currently defined**.

The result does not appear to be caused by an obvious split bug or by train/validation/test overflow. Instead, it appears that the model is learning useful discriminative patterns between normal and abnormal machine sounds under the present group-based evaluation setup.

This makes the all-machine baseline a strong and useful reference point for the next stage of the project.

### What this means for the next stage

Since the all-machine baseline now appears valid and strong, the project can move beyond basic pipeline construction.

The next stage should focus on:

- comparing final results across single-machine and all-machine baselines
- identifying baseline weaknesses, especially missed abnormal samples
- selecting controlled next experiments such as:
  - augmentation
  - alternative preprocessing
  - threshold tuning
  - class imbalance handling
  - stronger architectures
  - reconstruction-based methods such as autoencoders

This means the project is now shifting from **building the baseline pipeline** toward **using the baseline as a reference for experimentation**.


---