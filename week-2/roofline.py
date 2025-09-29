"""Arithmetic Intensity Sweep: CuPy version

Explores the memory-bound → compute-bound transition on H100
by varying the arithmetic intensity of a simple polynomial kernel.

Uses RawModule to load the kernel from roofline_kernel.cuh, ensuring
CUDA and CuPy implementations use identical device code.
"""

import numpy as np
import cupy as cp

# Load kernel from shared header file
# This ensures CUDA (roofline.cu) and CuPy use the exact same kernel
with open('roofline_kernel.cuh', 'r') as f:
    _kernel_code = f.read()
_module = cp.RawModule(code=_kernel_code)
_COMPUTE_K_KERNEL = _module.get_function('compute_k_terms')


def run_sweep(n: int, k: int, a: cp.ndarray, b: cp.ndarray):
    """Run warmup + timed iterations for given K value."""

    block = 256
    grid = (n + block - 1) // block

    # Arithmetic intensity calculation
    flops_per_element = 2.0 * k  # k additions + k multiplications
    bytes_per_element = 8.0      # 1 read (4B) + 1 write (4B)
    ai = flops_per_element / bytes_per_element

    print(f"\nK={k} (AI={ai:.3f} flops/byte):")

    # Warmup runs (silent)
    for w in range(3):
        _COMPUTE_K_KERNEL((grid,), (block,), (a, b, np.int32(n), np.int32(k)))
        cp.cuda.Stream.null.synchronize()

    # Timed runs
    nruns = 10
    times = []
    start = cp.cuda.Event()
    stop = cp.cuda.Event()

    for i in range(nruns):
        start.record()
        _COMPUTE_K_KERNEL((grid,), (block,), (a, b, np.int32(n), np.int32(k)))
        stop.record()
        stop.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, stop))

    # Statistics
    mean_ms = sum(times) / nruns

    # Calculate RMS for error checking
    sum_sq_diff = sum((t - mean_ms)**2 for t in times)
    rms = np.sqrt(sum_sq_diff / nruns)
    rms_percent = (rms / mean_ms) * 100.0

    mean_gflops = (n * flops_per_element / 1e9) / (mean_ms / 1e3)
    mean_bw = (n * bytes_per_element / 1e9) / (mean_ms / 1e3)
    percent_peak = 100.0 * mean_gflops / 67000.0

    print(f"  Mean: {mean_ms:.3f} ms ({mean_gflops:.2f} GFLOPS, {mean_bw:.2f} GB/s, {percent_peak:.2f}% peak)", end="")

    RMS_TOLERANCE = 5.0  # 5% tolerance
    if rms_percent > RMS_TOLERANCE:
        print(f" [WARNING: RMS={rms_percent:.2f}% > {RMS_TOLERANCE:.1f}%]")
    else:
        print()

    return ai, mean_gflops, mean_bw, percent_peak


if __name__ == "__main__":
    # Problem size: 2^26 elements (256 MB per array) for polynomial kernel
    n = 1 << 26  # 67M elements

    # Allocate and initialize
    a = cp.full(n, 1.5, dtype=cp.float32)
    b = cp.zeros(n, dtype=cp.float32)

    # Sweep through K values for polynomial kernel
    k_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    # Write CSV output
    with open("roofline_cupy.csv", "w") as f:
        f.write("k,ai,gflops,bandwidth,percent_peak\n")

        # Polynomial kernel sweep
        for k in k_values:
            ai, gflops, bw, pct_peak = run_sweep(n, k, a, b)
            f.write(f"{k},{ai:.6f},{gflops:.2f},{bw:.2f},{pct_peak:.2f}\n")
