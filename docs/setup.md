## Setup

> ⚠️ Make sure you have completed the setup steps before running the commands below:
> - [Setup guide](docs/setup.md)

### 1. Create and activate a virtual environment

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
````

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 2. Install PyTorch and torchaudio first

Install the PyTorch stack before installing the remaining project dependencies.

Important notes:

* `torch` and `torchaudio` must be installed as a **matching pair**
* do **not** install `torchaudio` separately from a different source after installing `torch`
* choose the install command from the official PyTorch selector for your platform and hardware

#### Recommended option: CPU-only

If you want the most reliable setup and do not specifically need GPU acceleration, use a CPU-only PyTorch install first.

This is the safest option for reproducing the project across different machines.

#### Optional option: GPU

If you want GPU training, install the PyTorch stack using the correct command for your machine from the official PyTorch selector.

Examples of hardware-specific options include:

* CUDA (NVIDIA GPU)
* ROCm
* Apple Silicon / MPS

Use the official selector to choose the correct command for your environment.

### 3. Install the remaining project dependencies

```bash
pip install -r requirements.txt
```

### 4. Linux audio backend note

On some Linux systems, `torchaudio.load(...)` may fail if the required audio backend libraries are not available.

If WAV loading fails, try:

```bash
pip install soundfile
sudo apt update
sudo apt install -y libsndfile1 ffmpeg
```

### 5. Recommended stable stack note

This project was tested with a pre-TorchCodec audio-loading path to avoid recent TorchAudio / TorchCodec environment issues.

If you encounter TorchCodec-related errors with newer TorchAudio versions, one practical workaround is to use a matching `torch` + `torchaudio` 2.8.x stack instead of moving immediately to TorchAudio 2.9+.

### 6. Download dataset

Download the MIMII dataset and place it under:

```text
data/raw/
```

The dataset must follow:

```text
data/raw/{snr_db}_{machine_type}/{machine_id}/{label}/*.wav
```

with `label` set to `normal` or `abnormal`.

---

## PyTorch / TorchAudio Troubleshooting

Common setup issues and what they usually mean:

### 1. `torchaudio` import or shared-library errors

Examples:

* missing `libcudart`
* missing CUDA runtime libraries
* import errors from `torchaudio` or `torchcodec`

Usually this means:

* `torch` and `torchaudio` do not match
* CPU and CUDA wheels were mixed
* `torchaudio` was installed separately after `torch` from a different source
* a newer TorchAudio version is trying to use TorchCodec

What to do:

* uninstall `torch`, `torchaudio`, `torchvision`, and `torchcodec`
* reinstall a matching PyTorch stack from one source
* avoid installing `torchaudio` separately afterward from plain PyPI if you already used the official PyTorch wheel index

### 2. `torchaudio.load(...)` cannot find an appropriate backend

Usually this means the Python package is installed but the system audio backend is missing.

On Linux, try:

```bash
pip install soundfile
sudo apt update
sudo apt install -y libsndfile1 ffmpeg
```

### 3. TorchCodec-related warnings or errors

TorchAudio is moving audio I/O toward TorchCodec in newer releases.

For this project, if the latest stack causes TorchCodec issues, it is acceptable to use a stable matching `torch` + `torchaudio` 2.8.x setup instead.

### 4. GPU not being used

Check:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

If `torch.cuda.is_available()` is `False`, your environment is not using CUDA correctly.

In that case:

* verify the correct CUDA PyTorch install command was used
* verify the NVIDIA driver / GPU environment on the machine
* or fall back to CPU-only setup


