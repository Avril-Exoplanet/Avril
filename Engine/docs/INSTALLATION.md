# Installation and device profiles

## Supported configurations

| Platform | Backend | Status | Recommended Python |
|---|---|---|---|
| Apple Silicon macOS | CPU | validated | 3.11 or 3.12 |
| Linux x86-64 | CPU | supported | 3.11 or 3.12 |
| Linux x86-64 + NVIDIA CUDA 12 | CUDA + CPU post-processing | validated | 3.11 or 3.12 |
| Windows 11 + WSL2 Ubuntu | CUDA + CPU post-processing | recommended Windows path | 3.11 or 3.12 |
| Native Windows | CPU; CUDA experimental | not release-validated | 3.11 or 3.12 |

## CPU/Mac

```bash
./setup_avril.sh cpu
source .venv/bin/activate
avril doctor --backend cpu
```

The installer deliberately keeps pip's small-file wheel cache on the native
system disk. This avoids build failures seen when a shell-level
`PIP_CACHE_DIR` points to ExFAT or a network volume. Set
`AVRIL_PIP_CACHE_DIR` only when the replacement is also a native local
filesystem.

## CUDA 12

Install an NVIDIA driver and the CUDA toolkit first. Avril compiles exact CUDA
kernels, so the runtime library alone is insufficient: `nvcc` must be present.

```bash
nvidia-smi
nvcc --version
./setup_avril.sh cuda
source .venv/bin/activate
avril doctor --backend cuda
```

Do not install `cupy` and `cupy-cuda12x` together. They conflict. Use the
version in `requirements/cuda12.txt`.

## Windows

WSL2 is the reproducible CUDA route:

1. enable WSL2 and install Ubuntu;
2. expose the NVIDIA GPU to WSL with the current Windows driver;
3. install the CUDA toolkit inside WSL, but do not install a second Linux GPU
   driver over the WSL interface;
4. clone/extract Avril inside the WSL filesystem, not `/mnt/c`, for faster
   compilation and small-file I/O;
5. run `./setup_avril.sh cuda`.

Native PowerShell setup is provided for CPU use and experimental native CUDA:

```powershell
.\setup_avril.ps1 cpu
.\.venv\Scripts\avril.exe doctor --backend cpu
```

## Data placement

Set a fast, capacious data directory before running large batches:

```bash
export AVRIL_DATA_DIR=/data/avril
```

The directory contains manifests, the resumable SPOC cache, run receipts, and
results. FITS files are not included in the source archive.
