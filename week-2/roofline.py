"""Arithmetic Intensity Sweep: CuPy version

Demonstrates roofline performance model using CuPy. Loads shared kernel from
roofline_kernel.cuh via RawModule (same kernel as CUDA version) and sweeps
polynomial degree K from 1 to 1000. See README.md for detailed explanation.
"""

import numpy as np
import cupy as cp

# KERNEL LOADING: Use CuPy RawModule to JIT-compile CUDA kernel
#
# RawModule workflow:
# 1. Read CUDA source code (roofline_kernel.cuh)
# 2. JIT-compile using NVRTC (NVIDIA Runtime Compiler)
# 3. Load compiled PTX/CUBIN into current GPU context
# 4. Extract kernel function by name
#
# This ensures the CUDA (roofline.cu) and CuPy implementations use
# IDENTICAL device code, enabling fair performance comparison.
with open('roofline_kernel.cuh', 'r') as f:
    _kernel_code = f.read()
_module = cp.RawModule(code=_kernel_code)
_COMPUTE_K_KERNEL = _module.get_function('compute_k_terms')


def run_sweep(n: int, k: int, a: cp.ndarray, b: cp.ndarray):
    """Measure kernel performance for a specific K value.

    This function implements GPU benchmarking best practices:

    1. WARMUP RUNS: Execute kernel 3 times before timing to eliminate:
       - JIT compilation overhead (NVRTC on first launch)
       - GPU frequency scaling effects
       - Cache cold-start effects

    2. MULTIPLE TIMED RUNS: Collect 10 timing measurements to:
       - Compute reliable mean performance
       - Detect timing variance (system interference)

    3. EVENT-BASED TIMING: Use CUDA events for GPU-side timing:
       - Eliminates CPU-GPU synchronization overhead
       - Provides microsecond precision
       - Correctly measures asynchronous kernel execution

    Parameters:
        n: Number of array elements
        k: Polynomial degree (controls arithmetic intensity)
        a: Input CuPy array (device memory)
        b: Output CuPy array (device memory)

    Returns:
        Tuple of (arithmetic_intensity, gflops, bandwidth, percent_peak)
    """

    # KERNEL LAUNCH CONFIGURATION:
    # Block size: 256 threads (typical for good occupancy on modern GPUs)
    # Grid size: Ceiling division to cover all elements
    block = 256
    grid = (n + block - 1) // block

    # ARITHMETIC INTENSITY CALCULATION:
    # Per element: k additions + k multiplications = 2k FLOPs
    # Memory access: 1 read (4 bytes) + 1 write (4 bytes) = 8 bytes
    # AI = FLOPs / Bytes = 2k / 8 = k/4 FLOPs/byte
    flops_per_element = 2.0 * k  # k additions + k multiplications
    bytes_per_element = 8.0      # 1 read (4B) + 1 write (4B)
    ai = flops_per_element / bytes_per_element

    print(f"\nK={k} (AI={ai:.3f} flops/byte):")

    # WARMUP PHASE: Run kernel 3 times to stabilize GPU state
    # First few launches may be slower due to:
    # - NVRTC JIT compilation (CuPy compiles kernels on first use)
    # - GPU power state transitions (boost clocks)
    # - Cache warming
    for w in range(3):
        # Launch kernel with grid/block dimensions and arguments
        # Arguments: (a, b, n, k) - must match kernel signature
        # Note: Python int must be converted to np.int32 for correct type
        _COMPUTE_K_KERNEL((grid,), (block,), (a, b, np.int32(n), np.int32(k)))
        cp.cuda.Stream.null.synchronize()  # Wait for completion

    # TIMED RUNS PHASE: Measure performance over 10 iterations
    nruns = 10
    times = []

    # Create CUDA events for precise GPU timing
    start = cp.cuda.Event()
    stop = cp.cuda.Event()

    for i in range(nruns):
        # Record start event in default stream
        start.record()

        # Launch kernel
        _COMPUTE_K_KERNEL((grid,), (block,), (a, b, np.int32(n), np.int32(k)))

        # Record stop event
        stop.record()
        stop.synchronize()  # Wait for stop event to complete

        # Compute elapsed time in milliseconds
        times.append(cp.cuda.get_elapsed_time(start, stop))

    # STATISTICS: Calculate mean and RMS deviation
    mean_ms = sum(times) / nruns

    # RMS (Root Mean Square) deviation measures timing stability
    # High RMS (>5%) indicates:
    # - System interference (other GPU workloads)
    # - Thermal throttling
    # - GPU frequency variations
    sum_sq_diff = sum((t - mean_ms)**2 for t in times)
    rms = np.sqrt(sum_sq_diff / nruns)
    rms_percent = (rms / mean_ms) * 100.0

    # PERFORMANCE METRICS:
    # GFLOPS = (Total FLOPs / 10^9) / (Time in seconds)
    mean_gflops = (n * flops_per_element / 1e9) / (mean_ms / 1e3)

    # Effective bandwidth = (Total bytes / 10^9) / (Time in seconds)
    mean_bw = (n * bytes_per_element / 1e9) / (mean_ms / 1e3)

    # Percentage of H100's theoretical peak FP32 performance (67 TFLOPS)
    percent_peak = 100.0 * mean_gflops / 67000.0

    print(f"  Mean: {mean_ms:.3f} ms ({mean_gflops:.2f} GFLOPS, {mean_bw:.2f} GB/s, {percent_peak:.2f}% peak)", end="")

    # Warn if timing variance exceeds 5% threshold
    RMS_TOLERANCE = 5.0  # 5% tolerance
    if rms_percent > RMS_TOLERANCE:
        print(f" [WARNING: RMS={rms_percent:.2f}% > {RMS_TOLERANCE:.1f}%]")
    else:
        print()

    return ai, mean_gflops, mean_bw, percent_peak


if __name__ == "__main__":
    # MAIN EXECUTION: Run roofline analysis sweep

    # Problem size: 2^26 elements = 67M elements
    # Array size: 67M * 4 bytes/float = 268 MB per array
    # - Large enough to avoid L2 cache effects (~40 MB on H100)
    # - Small enough for quick iteration
    n = 1 << 26  # 67M elements

    # Allocate and initialize GPU arrays
    # Input:  Fill with 1.01 (avoids overflow for K<1000)
    # Output: Zero-initialized (will be overwritten by kernel)
    a = cp.full(n, 1.01, dtype=cp.float32)
    b = cp.zeros(n, dtype=cp.float32)

    # K values sweep: Chosen to span memory-bound to compute-bound regions
    # - K=1:    AI=0.25 FLOPs/byte  (deeply memory-bound)
    # - K=50:   AI=12.5 FLOPs/byte  (near ridge point ~19.7)
    # - K=1000: AI=250 FLOPs/byte   (deeply compute-bound)
    k_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    # Write results to CSV for plotting
    with open("roofline_cupy.csv", "w") as f:
        # CSV header
        f.write("k,ai,gflops,bandwidth,percent_peak\n")

        # Run performance sweep across K values
        for k in k_values:
            ai, gflops, bw, pct_peak = run_sweep(n, k, a, b)

            # Write results: k, arithmetic_intensity, GFLOPS, bandwidth, percent_peak
            f.write(f"{k},{ai:.6f},{gflops:.2f},{bw:.2f},{pct_peak:.2f}\n")
