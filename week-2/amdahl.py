"""DataSci 211 Week 2 CuPy demo: Amdahl's Law multi-GPU scaling.

Demonstrates Amdahl's Law by measuring multi-GPU matrix multiplication speedup.
Includes serial data generation phase and parallel compute phase to show how
serial bottlenecks limit scalability. See README.md for detailed explanation
of Amdahl's Law concepts and methodology.

Usage:
    python amdahl.py           # Generate theoretical curves only
    python amdahl.py --measure # Include multi-GPU measurements
"""

import numpy as np
import cupy as cp


def amdahl(parallel_fraction: float, devices: int) -> float:
    """Calculate theoretical speedup using Amdahl's Law.

    Amdahl's Law formula:
        Speedup(p, N) = 1 / [(1-p) + p/N]

    Where:
        p = fraction of work that can be parallelized (0.0 to 1.0)
        N = number of parallel devices/processors
        (1-p) = serial fraction (bottleneck)

    Derivation:
        Total time = Serial time + Parallel time
        T(N) = T_serial + T_parallel/N
        T(N) = (1-p)T + pT/N
        Speedup = T(1) / T(N) = 1 / [(1-p) + p/N]

    Examples:
        amdahl(0.95, 4) = 3.48×  (95% parallel, 4 GPUs)
        amdahl(0.95, ∞) = 20×    (maximum speedup with p=0.95)
        amdahl(0.50, 4) = 1.6×   (50% parallel limits scaling)

    Parameters:
        parallel_fraction: Fraction of work that parallelizes (0.0-1.0)
        devices: Number of parallel processors/GPUs

    Returns:
        Theoretical speedup factor
    """
    return 1.0 / ((1 - parallel_fraction) + parallel_fraction / devices)


def measure_multi_gpu_matmul(M: int, N: int, K: int, num_gpus: int) -> float:
    """Measure matrix multiply across multiple GPUs with row-wise partitioning.

    This function demonstrates realistic multi-GPU scaling with both serial
    and parallel phases, illustrating Amdahl's Law in practice.

    WORKLOAD BREAKDOWN:
    1. SERIAL PHASE (GPU 0 only):
       - Data generation: Create random input data
       - This phase does NOT benefit from multiple GPUs
       - Simulates real-world serial bottlenecks (data loading, preprocessing)

    2. PARALLEL PHASE (all GPUs):
       - Partition matrix A row-wise across GPUs
       - Replicate matrix B on all GPUs
       - Each GPU computes its slice: C_i = A_i @ B
       - Speedup scales with number of GPUs (ideally)

    PARALLELIZATION STRATEGY:
    Matrix multiply: C[M×N] = A[M×K] @ B[K×N]

    - Split A into horizontal slices:
      GPU 0: A[0:M/N, :]
      GPU 1: A[M/N:2M/N, :]
      ...

    - Replicate B on all GPUs (needed by all slices)

    - Each GPU computes independent slice:
      C_i = A_i @ B (no inter-GPU communication needed)

    Parameters:
        M: Number of rows in A (partitioned across GPUs)
        N: Number of columns in B and C
        K: Inner dimension (shared)
        num_gpus: Number of GPUs to use (1-8)

    Returns:
        Elapsed time in milliseconds (includes serial + parallel phases)
    """
    # TIMING START: Begin measuring total execution time on GPU 0
    with cp.cuda.Device(0):
        start_main = cp.cuda.Event()
        start_main.record()

    # ==================================================================
    # SERIAL PHASE: Data generation on GPU 0 only
    # ==================================================================
    # This phase does NOT parallelize - represents the serial bottleneck
    #
    # Real-world serial bottlenecks include:
    # - Data loading from disk/network
    # - Data preprocessing (tokenization, normalization)
    # - Model initialization
    # - Logging/checkpointing
    #
    # We generate extra data (4× the matrix sizes) to increase the
    # serial fraction, targeting p ≈ 0.9 (10% serial, 90% parallel).
    # This demonstrates how even small serial fractions limit scaling.
    with cp.cuda.Device(0):
        # Generate random data on single GPU; by design,
        # this work does NOT benefit from additional GPUs
        total_elements = 4 * (M * K + K * N)
        serial_data = cp.random.rand(total_elements, dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()  # Wait for generation to complete

    # ==================================================================
    # PARALLEL PHASE: Multi-GPU matrix multiplication
    # ==================================================================
    # This phase DOES parallelize - each GPU works independently
    # Speedup depends on number of GPUs (ideally linear)

    # Calculate rows per GPU for partitioning
    rows_per_gpu = M // num_gpus

    # Allocate arrays and create streams on each GPU
    A_list = []  # Partitioned matrix (different slice per GPU)
    B_list = []  # Replicated matrix (same on all GPUs)
    C_list = []  # Output slices (different per GPU)
    streams = []  # Independent streams for concurrent execution

    for gpu_id in range(num_gpus):
        with cp.cuda.Device(gpu_id):
            # Create CUDA stream for asynchronous operations
            streams.append(cp.cuda.Stream())

            # Each GPU gets a horizontal slice of A
            # GPU i: rows [i*rows_per_gpu : (i+1)*rows_per_gpu]
            A_list.append(cp.random.rand(rows_per_gpu, K, dtype=cp.float32))

            # B is replicated on all GPUs (all slices need full B)
            B_list.append(cp.random.rand(K, N, dtype=cp.float32))

            # Allocate output slice on each GPU
            C_list.append(cp.empty((rows_per_gpu, N), dtype=cp.float32))

    # Synchronize all GPUs before starting parallel computation
    # Ensures setup is complete before timing the compute phase
    for gpu_id in range(num_gpus):
        with cp.cuda.Device(gpu_id):
            cp.cuda.Stream.null.synchronize()

    # Launch matrix multiplication on all GPUs concurrently
    # Each GPU computes: C_i = A_i @ B independently
    # No inter-GPU communication needed (embarrassingly parallel)
    for gpu_id in range(num_gpus):
        with cp.cuda.Device(gpu_id):
            with streams[gpu_id]:
                # Asynchronous matmul launch (kernels overlap across GPUs)
                cp.matmul(A_list[gpu_id], B_list[gpu_id], out=C_list[gpu_id])

    # Wait for all GPU compute streams to complete
    for gpu_id in range(num_gpus):
        with cp.cuda.Device(gpu_id):
            streams[gpu_id].synchronize()

    # TIMING END: Measure total elapsed time
    with cp.cuda.Device(0):
        stop_main = cp.cuda.Event()
        stop_main.record()
        stop_main.synchronize()
        elapsed_ms = cp.cuda.get_elapsed_time(start_main, stop_main)

    return elapsed_ms


if __name__ == "__main__":
    import sys

    # ==================================================================
    # CONFIGURATION: Check command-line arguments
    # ==================================================================
    # Usage:
    #   python amdahl.py          → Generate theoretical curves only
    #   python amdahl.py --measure → Include multi-GPU measurements
    run_measurements = '--measure' in sys.argv

    # ==================================================================
    # PART 1: Generate theoretical Amdahl's Law curves
    # ==================================================================
    # Compute theoretical speedup for various parallel fractions (p)
    # and GPU counts (N) to visualize Amdahl's Law behavior
    print("Generating Amdahl's Law speedup curves...")

    # Parallel fractions to explore
    parallel_fractions = [0.5, 0.75, 0.9, 0.95, 0.99]

    # GPU counts: powers of 2 up to 16
    # Sufficient to see speedup saturation for lower p values
    gpu_counts = [1, 2, 4, 8, 16]

    # Generate CSV with theoretical speedup data
    with open("amdahl_speedup.csv", "w") as f:
        f.write("parallel_fraction,num_gpus,speedup,max_speedup\n")

        for p in parallel_fractions:
            # Calculate theoretical maximum speedup (with infinite GPUs)
            # max_speedup = 1 / (1-p) = 1 / serial_fraction
            max_speedup = 1.0 / (1.0 - p)  # Theoretical ceiling

            for n_gpus in gpu_counts:
                # Calculate speedup for this (p, N) combination
                speedup = amdahl(p, n_gpus)
                f.write(f"{p},{n_gpus},{speedup:.6f},{max_speedup:.6f}\n")

    print(f"✓ Theoretical speedup data written to amdahl_speedup.csv")

    # ==================================================================
    # PART 2: Measure actual multi-GPU performance (optional)
    # ==================================================================
    if run_measurements:
        print("\n" + "="*60)
        print("Measuring actual multi-GPU matrix multiply performance...")
        print("="*60)

        # PROBLEM SIZE: Large enough to show clear Amdahl's Law behavior
        #
        # Total FLOPs: 2*M*N*K (matrix multiply complexity)
        # Problem breakdown:
        # - Serial part: Generate 4*(M*K + K*N) random elements on GPU 0
        # - Parallel part: Each GPU computes (M/num_gpus)×K @ K×N
        #
        # Chosen size (16384³):
        # - Large enough to keep GPUs busy (avoid scheduling overhead)
        # - Small enough for quick measurements (~seconds per run)
        M = 16384  # Total rows (partitioned across GPUs)
        N = 16384  # Columns in B and C
        K = 16384  # Inner dimension

        total_flops = 2 * M * N * K  # 2*M*N*K FLOPs for matmul
        print(f"Problem size: ({M}, {K}) @ ({K}, {N}) = {total_flops/1e12:.2f} TFLOPS")

        # GPU counts to test (limited by available hardware)
        # Adjust based on node configuration
        available_gpus = [1, 2, 4, 8]

        # WARMUP: Run small problem to initialize GPU state
        # - Loads CUDA libraries
        # - Initializes cuBLAS context
        # - Brings GPU to full clock speed
        print("\nWarming up...")
        measure_multi_gpu_matmul(1024, 1024, 1024, 1)

        # BASELINE MEASUREMENT: Single GPU performance
        # All speedup calculations relative to this baseline
        print("\nMeasuring baseline (1 GPU)...")
        baseline_times = []
        for _ in range(3):
            time_ms = measure_multi_gpu_matmul(M, N, K, 1)
            baseline_times.append(time_ms)

        # Use median to reduce impact of outliers
        baseline_ms = np.median(baseline_times)
        baseline_tflops = (total_flops / 1e12) / (baseline_ms / 1e3)
        print(f"  Baseline (1 GPU): {baseline_ms:.1f} ms = {baseline_tflops:.2f} TFLOPS (median of 3 runs)")

        # MULTI-GPU SCALING MEASUREMENTS
        # Measure performance with 1, 2, 4, 8 GPUs to observe scaling behavior
        results = []
        for num_gpus in available_gpus:
            print(f"\nMeasuring {num_gpus} GPU(s)...")
            times = []

            # Run 3 trials for each GPU count
            for run in range(3):
                time_ms = measure_multi_gpu_matmul(M, N, K, num_gpus)
                times.append(time_ms)
                tflops = (total_flops / 1e12) / (time_ms / 1e3)
                print(f"  Run {run+1}: {time_ms:.1f} ms = {tflops:.2f} TFLOPS")

            # Calculate metrics from median time
            median_ms = np.median(times)
            speedup = baseline_ms / median_ms  # Speedup relative to 1 GPU
            efficiency = speedup / num_gpus * 100  # Parallel efficiency %
            median_tflops = (total_flops / 1e12) / (median_ms / 1e3)

            print(f"  → Median: {median_ms:.1f} ms = {median_tflops:.2f} TFLOPS")
            print(f"  → Speedup: {speedup:.2f}×")
            print(f"  → Efficiency: {efficiency:.1f}%")

            # Store results for fitting
            results.append({
                'num_gpus': num_gpus,
                'time_ms': median_ms,
                'speedup': speedup,
                'efficiency': efficiency
            })

        # Write measured data to CSV
        with open("amdahl_measured.csv", "w") as f:
            f.write("num_gpus,time_ms,speedup,efficiency\n")
            for r in results:
                f.write(f"{r['num_gpus']},{r['time_ms']:.6f},{r['speedup']:.6f},{r['efficiency']:.6f}\n")

        print(f"\n✓ Measured speedup data written to amdahl_measured.csv")

        # ==================================================================
        # PART 3: Fit measured data to Amdahl's Law
        # ==================================================================
        # Find the parallel fraction 'p' that best explains measured speedups
        # This reveals the effective serial bottleneck in our workload
        if len(results) > 1:
            print("\n" + "="*60)
            print("Fitting measured data to Amdahl's Law...")
            print("="*60)

            # CURVE FITTING: Find p that minimizes sum of squared errors
            # between measured and predicted speedups
            #
            # Method: Grid search over possible p values
            # - Test p ∈ [0.5, 0.999] with 100 samples
            # - For each p, compute predicted speedup using Amdahl's Law
            # - Calculate squared error vs measured speedup
            # - Select p with minimum total error
            best_p = 0.0
            best_error = float('inf')

            for p_test in np.linspace(0.5, 0.999, 100):
                error = 0.0
                for r in results:
                    # Predicted speedup using Amdahl's Law with p_test
                    predicted = amdahl(p_test, r['num_gpus'])
                    # Squared error vs measured speedup
                    error += (predicted - r['speedup'])**2

                # Track best fit
                if error < best_error:
                    best_error = error
                    best_p = p_test

            # RESULTS: Report fitted parallel fraction
            # This tells us what fraction of our workload actually parallelized
            serial_fraction = 1.0 - best_p
            max_speedup = 1.0 / serial_fraction

            print(f"\n✓ Fitted parallel fraction: p = {best_p:.3f}")
            print(f"  Serial fraction: {serial_fraction:.3f} ({serial_fraction*100:.1f}%)")
            print(f"  Maximum theoretical speedup: {max_speedup:.1f}×")
            print(f"\nInterpretation:")
            print(f"  {serial_fraction*100:.1f}% of execution time is serial (doesn't parallelize)")
            print(f"  This limits speedup to at most {max_speedup:.1f}× regardless of GPU count")

            # Write fitted p to file for plotting
            with open("amdahl_fitted_p.txt", "w") as f:
                f.write(f"{best_p}\n")
    else:
        print("\nTo measure actual multi-GPU performance, run with --measure flag")
        print("  (requires GPU node with multiple GPUs available)")
