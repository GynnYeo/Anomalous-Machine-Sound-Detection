## Data Layout

Place the raw MIMII WAV files under `data/raw/`.

Expected folder layout:

```text
data/raw/{snr_db}_{machine_type}/{machine_id}/{label}/*.wav
```

Example:

```text
data/raw/-6_dB_fan/id_00/normal/*.wav
data/raw/-6_dB_fan/id_00/abnormal/*.wav
data/raw/0_dB_pump/id_02/normal/*.wav
data/raw/6_dB_valve/id_04/abnormal/*.wav
```

---

## Notes

* Valid labels are `normal` and `abnormal`
* The dataset structure encodes metadata (machine type, machine ID, SNR level, label)
* The indexing and split pipeline reads metadata directly from folder names, so this structure must be preserved
* `data/raw/` should contain the original, unmodified audio files

---

## Split manifests

All experiments in this repository are driven by **saved split manifests**.

Split manifests are stored under:

```text
data/splits/
```

Each manifest:

* defines train / validation / test membership
* is treated as a **source-of-truth artifact**
* ensures reproducibility across training, evaluation, and experiments

Example:

```text
data/splits/fan_split_seed42.csv
data/splits/all_machines_split_seed42.csv
```

---

## Split strategy

Splits are created using **group-based partitioning**, not file-level random splits.

For one-machine-type experiments:

* groups are defined as:

```text
group_id = (machine_id, snr_db)
```

This ensures:

* all recordings from the same machine and SNR condition stay in the same split
* leakage between train, validation, and test is minimized

For multi-machine experiments:

* each machine type is split independently by groups
* the results are then concatenated into a single manifest

---

## Why manifests matter

Using saved manifests allows:

* reproducible experiments across runs
* fair comparison between model variants (baseline, deeper, wider)
* consistent evaluation across hyperparameter tuning and threshold tuning
* decoupling of data splitting from training logic

All training, evaluation, grid search, and threshold tuning workflows rely on the same manifest input.

---

## Best practices

* Do not modify files inside `data/raw/` after downloading
* Do not regenerate splits unless intentionally creating a new experiment
* Always keep the manifest used for a reported result
* Treat manifests as part of your experiment artifacts

When reproducing a result, always use the exact same:

* manifest file
* model configuration
* checkpoint
