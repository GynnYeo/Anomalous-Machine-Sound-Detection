## Setup

Follow the steps below to set up the environment before running any training or evaluation commands.

---

### 1. Create and activate a virtual environment

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

---

### 2. Install PyTorch and TorchAudio

Install the PyTorch stack **before** installing the remaining project dependencies.

Important notes:

* `torch` and `torchaudio` must be installed as a **matching pair**
* do **not** install `torchaudio` separately afterward from a different source
* use the official PyTorch selector to choose the correct installation command

#### Recommended option: CPU-only

For maximum compatibility and reproducibility across machines, use a CPU-only installation.

#### Optional option: GPU

If GPU acceleration is required, install the appropriate PyTorch build for your system:

* CUDA (NVIDIA GPU)
* ROCm
* Apple Silicon / MPS

Use the official PyTorch selector to obtain the correct command.

---

### 3. Install project dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Download and place the dataset

Download the MIMII dataset and place it under:

```text
data/raw/
```

The dataset must follow this structure:

```text
data/raw/{snr_db}_{machine_type}/{machine_id}/{label}/*.wav
```

where:

* `{label}` is either `normal` or `abnormal`

---

### 5. Linux audio backend note

On some Linux systems, audio loading may fail if backend libraries are missing.

If `torchaudio.load(...)` fails, install:

```bash
pip install soundfile
sudo apt update
sudo apt install -y libsndfile1 ffmpeg
```

---

### 6. Recommended stable stack note

This project was tested using a pre-TorchCodec audio-loading path.

If you encounter issues with newer TorchAudio versions:

* consider using a matching `torch` + `torchaudio` 2.8.x stack
* avoid mixing versions from different sources

---

## PyTorch / TorchAudio Troubleshooting

### 1. `torchaudio` import or shared-library errors

Examples:

* missing `libcudart`
* missing CUDA runtime libraries
* import errors from `torchaudio` or `torchcodec`

These usually indicate:

* mismatched `torch` and `torchaudio` versions
* mixing CPU and CUDA builds
* installing `torchaudio` separately after `torch`
* incompatibilities with TorchCodec

**Fix:**

* uninstall `torch`, `torchaudio`, `torchvision`, and `torchcodec`
* reinstall using a single consistent source

---

### 2. Audio backend not found

If `torchaudio.load(...)` cannot find a backend:

```bash
pip install soundfile
sudo apt update
sudo apt install -y libsndfile1 ffmpeg
```

---

### 3. TorchCodec-related issues

TorchAudio is transitioning to TorchCodec in newer versions.

If errors occur:

* use a stable matching `torch` + `torchaudio` 2.8.x setup
* avoid upgrading blindly to newer versions

---

### 4. GPU not being used

Check:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

If `torch.cuda.is_available()` returns `False`:

* verify the correct PyTorch installation command
* check GPU drivers and environment
* or fall back to CPU-only setup

---

## Notes

* The repository is designed to work consistently across CPU and GPU environments
* The main training and evaluation workflows are script-driven (`scripts/`), not notebook-driven
* Additional experiment utilities (grid search and threshold tuning) use the same environment and dependencies as the baseline pipeline
