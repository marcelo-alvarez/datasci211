# DataSci 211 — Week 2 Student Guide

Use this README to get ready for Lecture 2 and validate your GPU workflow on Marlowe.

---

## 1. Prerequisites
- SUNet ID with access to Slurm account `marlowe-c001`.
- Terminal with SSH (macOS/Linux shell; Windows via WSL or VS Code Remote SSH).
- GitHub account for cloning this repository.

## 2. Clone and enter the week 2 kit
```bash
git clone https://github.com/marcelo-alvarez/datasci211-course-materials.git
cd datasci211-course-materials/week-2
```
All commands below assume you remain inside `week-2/` unless noted. Warm-up CPU/GPU examples live in [`week-1`](../week-1/README.md).

## 3. Python environment (shared installation)
A shared Micromamba environment has been installed at `/scratch/c001/sw/micromamba` with the `ds211-python` environment containing Python 3.11, CuPy, matplotlib, and NumPy.

To activate this environment in your shell:
```bash
export MAMBA_ROOT_PREFIX="/scratch/c001/sw/micromamba"
eval "$("$MAMBA_ROOT_PREFIX/bin/micromamba" shell hook --shell=bash)"
micromamba activate ds211-python
```

> **Note:** If you need to install the environment yourself (e.g., for a different location), run:
> ```bash
> bash setup_shared_micromamba.sh
> ```
> and edit the `INSTALL_BASE` variable at the top of the script.

## 4. Interactive GPU session on Marlowe
Request a GPU allocation and activate the environment:
```bash
salloc -A marlowe-c001 -p class --qos=class --gpus=1 --cpus-per-task=8 --mem=32G --time=01:00:00

cd ~/datasci211-course-materials/week-2
module purge
module load nvhpc
cuda_prefix="${NVHPC_ROOT}/cuda"
export CUDA_HOME="$cuda_prefix"
export CUDA_PATH="$cuda_prefix"
export CPATH="$cuda_prefix/include:${CPATH:-}"
export MAMBA_ROOT_PREFIX="/scratch/c001/sw/micromamba"
eval "$("$MAMBA_ROOT_PREFIX/bin/micromamba" shell hook --shell=bash)"
micromamba activate ds211-python
```

Once `salloc` grants the allocation you are already running on the compute node; use `hostname` to confirm.

## 5. Run roofline analysis
The roofline model demonstrates GPU performance characteristics by varying arithmetic intensity (FLOPs per byte).

`run_roofline.sh` runs both CUDA and CuPy arithmetic intensity sweeps via `srun`, then generates a roofline plot locally (no GPU allocation needed for plotting):
```bash
cd ~/datasci211-course-materials/week-2
bash run_roofline.sh
```
This will:
- Request GPU allocation via `srun`
- Compile `roofline.cu` and run the CUDA sweep (uses shared kernel from `roofline_kernel.cuh`)
- Run the CuPy sweep from `roofline.py` (JIT-compiles the same shared kernel)
- Generate CSV files: `roofline_cuda.csv` and `roofline_cupy.csv`
- Create roofline plot: `roofline_h100.png`

The plot shows memory-bound vs compute-bound regions and demonstrates that CUDA and CuPy achieve identical performance.

## 6. Run Amdahl's Law multi-GPU scaling
Amdahl's Law shows how serial bottlenecks limit parallel speedup. This example uses multi-GPU matrix multiplication.

`run_amdahl.sh` generates theoretical Amdahl curves and optionally measures actual multi-GPU performance:
```bash
cd ~/datasci211-course-materials/week-2

# Generate theoretical curves only (runs on login node):
bash run_amdahl.sh

# Include multi-GPU measurements (requires 8 GPUs):
bash run_amdahl.sh --measure
```

With `--measure`, this will:
- Request 8 GPU allocation via `srun`
- Run `amdahl.py` which partitions matrix multiply across GPUs
- Measure speedup at 1, 2, 4, 8 GPUs
- Fit measured data to Amdahl's Law to find effective parallel fraction
- Generate CSV files: `amdahl_speedup.csv`, `amdahl_measured.csv`, `amdahl_fitted_p.txt`
- Create plot: `amdahl_plot.png`

The code demonstrates realistic parallel scaling with serial data generation overhead (~10%) limiting maximum speedup to ~10× despite using 8 GPUs.

## 7. Troubleshooting tips
- `micromamba: command not found` → ensure `/scratch/c001/sw/micromamba` exists and you've set `MAMBA_ROOT_PREFIX` correctly.
- `python` resolves to `/usr/bin/python` → run the Micromamba shell hook before `micromamba activate`.
- `ImportError: cupy` → the shared environment should have CuPy pre-installed. Check `micromamba list` after activating.
- `Invalid qos specification` → confirm `--partition=class --qos=class` and that you have no other class QoS job active.

Successfully running both starter examples and the CuPy demo means you are ready for Lecture 2.
