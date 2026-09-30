NVIDIA VFX Python Samples
=========================

This repository contains sample applications for the nvidia-vfx Python bindings of the NVIDIA Video Effects (VFX) SDK. The Video Super Resolution sample is a Python reference application that demonstrates real-time video upscaling: it reads an input video file, upscales each frame using GPU-accelerated AI models to improve sharpness and reduce compression artifacts, and writes the enhanced result to a new output file based on command-line options. The Video Frame Generation sample generates intermediate frames between consecutive input frames for smoother playback or slow motion. The TrueHDR sample converts an SDR video to HDR10 (BT.2020 / ST.2084) using GPU-accelerated AI models and writes an HEVC Main10 output file with the correct color metadata.

Requirements
------------

- **Python:** 3.10 or later
- **GPU:** NVIDIA GPU with Tensor Cores
  - Video Super Resolution: Turing, Ampere, Ada, Blackwell, or Hopper architecture
  - Video Frame Generation: Ada or Blackwell architecture
  - TrueHDR: Turing, Ampere, Ada, Blackwell, or Hopper architecture
- **GPU driver:**
  - Windows x64: 570.65 or later (for Tesla Compute Cluster (TCC) devices, 595 or later is required)
  - NVIDIA RTX Spark (Windows on ARM): 616.41 or later
  - Linux: 570.190+, 580.82+, or 590.44+
- **Git:** https://git-scm.com/downloads
- **Git LFS:** https://git-lfs.com/ (required for sample video assets)
- **OS:**
  - Windows: 64-bit Windows 10 or later on x64, or NVIDIA RTX Spark on Windows on ARM
  - Linux: Ubuntu 20.04, 22.04, 24.04, Debian 12, or RHEL 8/9

Setup
-----

### Clone the repository

```bash
git clone git@github.com:NVIDIA-Maxine/nvidia-vfx-python-samples.git        # Using SSH, or
# git clone https://github.com/NVIDIA-Maxine/nvidia-vfx-python-samples.git  # Using HTTP
cd nvidia-vfx-python-samples
```

Initialize Git LFS so sample videos are downloaded correctly:

```bash
git lfs install
git lfs pull
```

### Create a virtual environment and install dependencies

#### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

#### Windows x64

> Create the venv at a short path near a drive root (e.g. `C:\dev\vfx\.venv`).
> PyTorch's deeply nested files can otherwise hit Windows' path-length limit.

```powershell
$ErrorActionPreference = "Stop"

python -m venv C:\dev\vfx\.venv
& "C:\dev\vfx\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

#### NVIDIA RTX Spark

Use a native ARM64 Python 3.11, 3.12, 3.13, or 3.14 installation.
The requirements file downloads the pinned PyTorch 2.14 CUDA 13.4 release from
the NVIDIA package index, so internet access is required during installation.

> Create the venv at a short path near a drive root (e.g. `C:\dev\vfx\.venv`).
> PyTorch's deeply nested files can otherwise hit Windows' path-length limit.

```powershell
$ErrorActionPreference = "Stop"

python -m venv C:\dev\vfx\.venv
& "C:\dev\vfx\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r requirements-rtx-spark.txt
python -m pip check
```

Video Super Resolution Sample
-----------------------------

From the repository root (with your virtual environment activated):

```bash
# Default: 2× upscale, HIGH quality, using bundled sample video
python video_super_resolution.py

# Custom input and output
python video_super_resolution.py -i path/to/input.mp4 -o path/to/output.mp4

# 4× upscale with ULTRA quality
python video_super_resolution.py --scale 4 --quality ULTRA

# Same-resolution denoise (no upscaling)
python video_super_resolution.py --scale 1 --quality DENOISE_HIGH

# Streaming upscale with reduced effect strength
python video_super_resolution.py --quality STREAMING_MEDIUM --strength 0.75

# Maximum-quality streaming upscale
python video_super_resolution.py --quality STREAMING_ULTRA
```

Video Super Resolution Sample — Command-Line Reference
------------------------------------------------------

| Argument            | Description |
|---------------------|-------------|
| `-i`, `--input`     | The input video file path. Default: `assets/Drift_RUN_Master_Custom.mp4` (relative to the script location) |
| `-o`, `--output`    | The output video file path. Default: `output/sample_sr.mp4` (relative to the script location) |
| `--scale`          | The scale factor. Choices: `1`, `2`, `3`, `4`. Default: `2`. Values `2`, `3`, and `4` upscale both width and height (for example, 1920x1080 → 3840x2160 at `2`×), with higher factors demanding more GPU memory and compute. A value of `1` keeps the original resolution and enables same‑resolution cleanup modes (denoise/deblur) instead of upscaling. |
| `--quality`        | The processing mode / quality level. For `--scale` ≥ `2`, choices: `BICUBIC`, `LOW`, `MEDIUM`, `HIGH`, `ULTRA` (default: `HIGH`), where `BICUBIC` is a fast non-AI baseline, `LOW`/`MEDIUM` favor speed, and `HIGH`/`ULTRA` maximize detail enhancement and artifact reduction. Streaming modes: `STREAMING_MEDIUM` (21), `STREAMING_ULTRA` (23) for low-latency real-time upscaling. When `--scale 1` is used, additional choices: `DENOISE_LOW`, `DENOISE_MEDIUM`, `DENOISE_HIGH`, `DENOISE_ULTRA` (noise/compression artifact removal) and `DEBLUR_LOW`, `DEBLUR_MEDIUM`, `DEBLUR_HIGH`, `DEBLUR_ULTRA` (sharpening for soft or blurry footage). |
| `--strength`       | Effect strength in `[0.0, 1.0]`. Default: `1.0`. Lower values reduce the effect contribution in the final output. |
| `-h`, `--help`      | Display help information for the command. |

Video Frame Generation Sample
-----------------------------

From the repository root (with your virtual environment activated):

```bash
# Default: double the input frame rate using bundled sample video
python video_frame_generation.py

# High-quality frame generation
python video_frame_generation.py --mode HIGH

# Generate three evenly spaced frames between each input pair
python video_frame_generation.py --multiplier 4

# Generate frames at explicit temporal positions
python video_frame_generation.py --timesteps 0.25,0.5,0.75

# Write a 24 FPS slow-motion video
python video_frame_generation.py --multiplier 4 --target-fps 24

# Process a 10-bit source and preserve 10-bit HEVC output
python video_frame_generation.py -i path/to/10bit_input.mp4 -o output/vfg_10bit.mp4
```

The sample detects 10-bit input pixel formats automatically. It decodes them without
reducing to 8-bit, packs frames as RGB10A2 for Video Frame Generation, and writes
HEVC Main10 output. Eight-bit sources continue to use the existing RGB8 path.

Video Frame Generation Sample — Command-Line Reference
------------------------------------------------------

| Argument            | Description |
|---------------------|-------------|
| `-i`, `--input`     | The input video file path. Eight-bit and 10-bit sources are detected automatically. Default: `assets/Drift_RUN_Master_Custom.mp4` (relative to the script location) |
| `-o`, `--output`    | The output video file path. Ten-bit input produces HEVC Main10 output. Default: `output/sample_vfg.mp4` (relative to the script location) |
| `--multiplier`      | The frame-rate multiplier. Default: `2`. The sample generates `multiplier - 1` evenly spaced frames between each consecutive input frame, then writes the original current frame. Cannot be combined with `--timesteps`. |
| `--timesteps`       | A comma-separated list of temporal positions in the open interval `(0.0, 1.0)`, such as `0.25,0.5,0.75`. A value of `0.5` generates a frame halfway between the two input frames. Cannot be combined with `--multiplier`. |
| `--target-fps`      | The output frame rate. Default: `0`, which uses the input frame rate multiplied by the number of frames written per input pair. Set a lower value for slow motion. |
| `--mode`            | The processing mode. Choices: `LOW`, `MEDIUM`, `HIGH`. Default: `MEDIUM`. |
| `--automatic-shot-change-detection`, `--no-automatic-shot-change-detection` | Enable or disable automatic shot-change detection. Default: enabled. |
| `-h`, `--help`      | Display help information for the command. |

TrueHDR Sample
--------------

From the repository root (with your virtual environment activated):

```bash
# Default: convert bundled SDR sample to HDR10
python true_hdr.py

# Custom input and output
python true_hdr.py -i path/to/sdr_input.mp4 -o path/to/hdr10_output.mp4

# Target a 1000-nit display with slightly boosted saturation
python true_hdr.py --luminance 1000 --saturation 120

# Skip the DL-Debander pass (faster; may show banding on gradients)
python true_hdr.py --debanding-off
```

The sample accepts any SDR video PyAV can decode (RGB8 pipeline). It runs
TrueHDR to produce packed RGB10A2 output, wraps that in an FFmpeg
`x2bgr10le` frame, and writes an HEVC Main10 (`yuv420p10le`) file tagged
with HDR10 signalling: BT.2020 primaries, ST.2084 (PQ) transfer, BT.2020
non-constant luminance matrix, limited range. `hevc_nvenc` is preferred;
`libx265` is used as a fallback.

If you need to inspect the raw TrueHDR output rather than encoding it,
`video_io_utils.py` includes an `unpack_rgb10a2()` helper that turns the
packed `(H, W)` uint32 words into per-channel R/G/B/A planes. Import it
as `from video_io_utils import unpack_rgb10a2`; pass either a PyTorch
tensor or a NumPy array and it returns the matching type.

TrueHDR Sample — Command-Line Reference
---------------------------------------

| Argument            | Description |
|---------------------|-------------|
| `-i`, `--input`     | The input SDR video file path. Default: `assets/Drift_RUN_Master_Custom.mp4` (relative to the script location) |
| `-o`, `--output`    | The output HDR10 (HEVC Main10) video file path. Default: `output/sample_truehdr.mp4` (relative to the script location) |
| `--contrast`        | Contrast tunable. Range: `0..200`. Default: `100`. Values above `100` steepen the tone curve; values below `100` flatten it. |
| `--saturation`      | Saturation tunable. Range: `0..200`. Default: `100`. Values above `100` boost color saturation; `0` produces a neutral gray. |
| `--middle-gray`     | Middle-gray reference. Range: `10..100`. Default: `50`. Controls where the SDR middle-gray point maps on the HDR tone curve. |
| `--luminance`       | Target HDR display peak luminance in nits. Range: `400..2000`. Default: `650`. Match this to your target display for best results (for example, `1000` for a 1000-nit HDR monitor). |
| `--debanding-off`   | Skip the DL-Debander pass. Debanding is on by default. |
| `-h`, `--help`      | Display help information for the command. |

Links
-----

- [nvidia-vfx Python package on PyPI](https://pypi.org/project/nvidia-vfx/)
- [nvidia-vfx Python bindings guide](https://docs.nvidia.com/maxine/vfx-python/latest/index.html)

> **Note:** This project is currently not accepting contributions.
