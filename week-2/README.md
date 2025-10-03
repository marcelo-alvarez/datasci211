# DataSci 211 — Week 2 Student Guide

Use this README to validate your GPU workflow on Marlowe and work through the two examples provided.

---

## 1. Prerequisites
- SUNet ID with access to Marlowe Slurm account `marlowe-c001`.
- Terminal with SSH or Open OnDemand with Code Server tool.
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

### Overview
The roofline model is a visual performance characterization that plots achieved performance (GFLOPS) vs arithmetic intensity (FLOPs/byte). This analysis demonstrates GPU performance characteristics by varying arithmetic intensity.

**1. ROOFLINE MODEL:**
- Performance is limited by the minimum of:
  - Memory bandwidth ceiling: Performance = Bandwidth × AI
  - Compute throughput ceiling: Performance = Peak FLOPS

**2. ARITHMETIC INTENSITY (AI):**
- AI = FLOPs / Bytes accessed from memory
- Low AI (<ridge point): Memory-bound, limited by bandwidth
- High AI (>ridge point): Compute-bound, limited by FLOPS

**3. RIDGE POINT:**
- AI_ridge = Peak FLOPS / Peak Bandwidth
- For H100: ~67 TFLOPS / 3.4 TB/s ≈ 19.7 FLOPs/byte
- This is where the workload transitions from memory- to compute-bound

**4. CUPY RAWMODULE:**
- CuPy's RawModule allows loading CUDA C/C++ kernels directly into Python
- Enables using optimized CUDA kernels from Python
- Allows fair comparison between CUDA and CuPy (same kernel)

### Implementation

**Polynomial Kernel (`roofline_kernel.cuh`):**
- Computes: `b[i] = a[i] + a[i]^2 + ... + a[i]^k`
- FLOPs: K additions + K multiplications = 2K FLOPs per element
- Memory: 1 read (4 bytes) + 1 write (4 bytes) = 8 bytes per element
- Arithmetic intensity: AI = 2K / 8 = K/4 FLOPs/byte
- By varying K from 1 to 1000, we sweep from memory-bound to compute-bound

**Performance Regions:**
- Small K (low AI): Memory-bound, limited by bandwidth (~3.4 TB/s on H100)
- Large K (high AI): Compute-bound, limited by FP32 throughput (~67 TFLOPS on H100)

### Running

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

### Overview
Amdahl's Law predicts the theoretical speedup achievable when parallelizing a workload. It shows how even small serial fractions can severely limit parallel scalability.

**1. AMDAHL'S LAW:**
```
Speedup = 1 / [(1-p) + p/N]
```
where:
- `p` = parallel fraction (0 to 1)
- `N` = number of processors (GPUs)
- `(1-p)` = serial fraction

The serial fraction limits maximum speedup
- Example: 10% serial (p=0.9) → max speedup = 10×, even with infinite GPUs

**2. MULTI-GPU SCALING:**
Any workload has serial components, e.g.:
- Data generation/loading 
- Model synchronization
- I/O operations (logging, checkpointing)

This example uses matrix multiplication partitioned across GPUs with (mock) serial data generation to show the scaling behavior.

**3. PARALLEL EFFICIENCY:**
```
Efficiency = Speedup / N
```
- Perfect: 100% (linear scaling)
- Good: >80%
- Limited: <50%

### Implementation Details

**Multi-GPU Matrix Multiply (`amdahl.py`):**

**Workload breakdown:**
1. **Serial phase (GPU 0 only):**
   - Data generation: Create random input data
   - Does NOT benefit from multiple GPUs
   - Simulates real-world serial bottlenecks

2. **Parallel phase (all GPUs):**
   - Partition matrix A row-wise across GPUs
   - Replicate matrix B on all GPUs
   - Each GPU computes its slice: C_i = A_i @ B independently
   - Speedup scales with number of GPUs (ideally)

**Parallelization strategy:**
- Matrix multiply: C[M×N] = A[M×K] @ B[K×N]
- Split A into horizontal slices (GPU i gets rows `[i*M/N : (i+1)*M/N]`)
- Replicate B on all GPUs (all slices need full B)
- Each GPU computes independent slice (no inter-GPU communication needed)

**Why this demonstrates Amdahl's Law:**
- Serial phase: Fixed cost regardless of GPU count
- Parallel phase: Scales linearly with GPU count
- Measured speedup should match Amdahl's Law with fitted p

### Running the Analysis

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

The code demonstrates example parallel scaling with serial data generation overhead (~10%) limiting speedup to ~4x even on 8 GPUs.
