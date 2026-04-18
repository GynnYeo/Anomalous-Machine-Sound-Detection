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

Notes:

* Valid labels are `normal` and `abnormal`
* The split/index pipeline reads metadata from the folder names, so keep this structure unchanged
* `data/raw/` should contain the original audio files
* Saved split manifests are written under `data/splits/`