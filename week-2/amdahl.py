"""DataSci 211 Week 2 CuPy demo: Amdahl's Law multi-GPU scaling."""

import numpy as np
import cupy as cp


def amdahl(parallel_fraction: float, devices: int) -> float:
    return 1.0 / ((1 - parallel_fraction) + parallel_fraction / devices)


def measure_multi_gpu_matmul(M: int, N: int, K: int, num_gpus: int) -> float:
    """Measure matrix multiply across multiple GPUs with row-wise partitioning.

    Computes C = A @ B where A is (M, K) and B is (K, N).
    Includes serial data generation phase and parallel compute phase.
    A is partitioned row-wise across GPUs, B is replicated on each GPU.

    Args:
        M: Number of rows in A (partitioned across GPUs)
        N: Number of columns in B
        K: Inner dimension
        num_gpus: Number of GPUs to use (1-8)

    Returns:
        Elapsed time in milliseconds
    """
    # Start timing on GPU 0
    with cp.cuda.Device(0):
        start_main = cp.cuda.Event()
        start_main.record()

    # SERIAL PHASE: Data generation on GPU 0 only (doesn't parallelize)
    # This simulates real workloads where data loading/preprocessing is sequential
    with cp.cuda.Device(0):
        # Generate input data on single GPU
        # Increase serial work to target p ~ 0.9 (10% serial, 90% parallel)
        # Generate 4x the data to increase serial fraction
        total_elements = 4 * (M * K + K * N)
        serial_data = cp.random.rand(total_elements, dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()

    # PARALLEL PHASE: Partition A row-wise, replicate B, compute
    rows_per_gpu = M // num_gpus

    # Create arrays and streams on each GPU
    A_list = []
    B_list = []
    C_list = []
    streams = []

    for gpu_id in range(num_gpus):
        with cp.cuda.Device(gpu_id):
            streams.append(cp.cuda.Stream())
            # Each GPU gets a horizontal slice of A
            A_list.append(cp.random.rand(rows_per_gpu, K, dtype=cp.float32))
            # B is replicated on all GPUs
            B_list.append(cp.random.rand(K, N, dtype=cp.float32))
            # Each GPU produces a horizontal slice of C
            C_list.append(cp.empty((rows_per_gpu, N), dtype=cp.float32))

    # Synchronize all GPUs before parallel compute
    for gpu_id in range(num_gpus):
        with cp.cuda.Device(gpu_id):
            cp.cuda.Stream.null.synchronize()

    # Launch matmul on all GPUs in parallel
    for gpu_id in range(num_gpus):
        with cp.cuda.Device(gpu_id):
            with streams[gpu_id]:
                cp.matmul(A_list[gpu_id], B_list[gpu_id], out=C_list[gpu_id])

    # Wait for all GPU streams to complete
    for gpu_id in range(num_gpus):
        with cp.cuda.Device(gpu_id):
            streams[gpu_id].synchronize()

    # Stop timing on GPU 0
    with cp.cuda.Device(0):
        stop_main = cp.cuda.Event()
        stop_main.record()
        stop_main.synchronize()
        elapsed_ms = cp.cuda.get_elapsed_time(start_main, stop_main)

    return elapsed_ms


if __name__ == "__main__":
    import sys

    # Check if we should run measurements or just generate theoretical curves
    run_measurements = '--measure' in sys.argv

    # Part 1: Generate theoretical Amdahl's Law curves
    print("Generating Amdahl's Law speedup curves...")

    # Parallel fractions to explore
    parallel_fractions = [0.5, 0.75, 0.9, 0.95, 0.99]

    # GPU counts: powers of 2 up to 16
    gpu_counts = [1, 2, 4, 8, 16]

    # Generate theoretical data
    with open("amdahl_speedup.csv", "w") as f:
        f.write("parallel_fraction,num_gpus,speedup,max_speedup\n")

        for p in parallel_fractions:
            max_speedup = 1.0 / (1.0 - p)  # Theoretical ceiling
            for n_gpus in gpu_counts:
                speedup = amdahl(p, n_gpus)
                f.write(f"{p},{n_gpus},{speedup:.6f},{max_speedup:.6f}\n")

    print(f"✓ Theoretical speedup data written to amdahl_speedup.csv")

    # Part 2: Measure actual multi-GPU performance
    if run_measurements:
        print("\n" + "="*60)
        print("Measuring actual multi-GPU matrix multiply performance...")
        print("="*60)

        # Problem size: Large enough to show clear Amdahl's Law behavior
        # Serial part: Generate M*K + K*N elements on single GPU
        # Parallel part: Each GPU computes (M/num_gpus, K) @ (K, N)
        M = 16384  # Total rows (partitioned across GPUs)
        N = 16384  # Columns in B and C
        K = 16384  # Inner dimension

        total_flops = 2 * M * N * K  # 2*M*N*K for matmul
        print(f"Problem size: ({M}, {K}) @ ({K}, {N}) = {total_flops/1e12:.2f} TFLOPS")

        # Available GPU counts on this node
        available_gpus = [1, 2, 4, 8]

        # Warmup
        print("\nWarming up...")
        measure_multi_gpu_matmul(1024, 1024, 1024, 1)

        # Measure baseline (1 GPU)
        print("\nMeasuring baseline (1 GPU)...")
        baseline_times = []
        for _ in range(3):
            time_ms = measure_multi_gpu_matmul(M, N, K, 1)
            baseline_times.append(time_ms)
        baseline_ms = np.median(baseline_times)
        baseline_tflops = (total_flops / 1e12) / (baseline_ms / 1e3)
        print(f"  Baseline (1 GPU): {baseline_ms:.1f} ms = {baseline_tflops:.2f} TFLOPS (median of 3 runs)")

        # Measure multi-GPU performance
        results = []
        for num_gpus in available_gpus:
            print(f"\nMeasuring {num_gpus} GPU(s)...")
            times = []
            for run in range(3):
                time_ms = measure_multi_gpu_matmul(M, N, K, num_gpus)
                times.append(time_ms)
                tflops = (total_flops / 1e12) / (time_ms / 1e3)
                print(f"  Run {run+1}: {time_ms:.1f} ms = {tflops:.2f} TFLOPS")

            median_ms = np.median(times)
            speedup = baseline_ms / median_ms
            efficiency = speedup / num_gpus * 100
            median_tflops = (total_flops / 1e12) / (median_ms / 1e3)

            print(f"  → Median: {median_ms:.1f} ms = {median_tflops:.2f} TFLOPS")
            print(f"  → Speedup: {speedup:.2f}×")
            print(f"  → Efficiency: {efficiency:.1f}%")

            results.append({
                'num_gpus': num_gpus,
                'time_ms': median_ms,
                'speedup': speedup,
                'efficiency': efficiency
            })

        # Write measured data
        with open("amdahl_measured.csv", "w") as f:
            f.write("num_gpus,time_ms,speedup,efficiency\n")
            for r in results:
                f.write(f"{r['num_gpus']},{r['time_ms']:.6f},{r['speedup']:.6f},{r['efficiency']:.6f}\n")

        print(f"\n✓ Measured speedup data written to amdahl_measured.csv")

        # Fit to find effective parallel fraction
        if len(results) > 1:
            # Simple fitting: find p that minimizes error
            best_p = 0.0
            best_error = float('inf')

            for p_test in np.linspace(0.5, 0.999, 100):
                error = 0.0
                for r in results:
                    predicted = amdahl(p_test, r['num_gpus'])
                    error += (predicted - r['speedup'])**2
                if error < best_error:
                    best_error = error
                    best_p = p_test

            print(f"\n✓ Fitted parallel fraction: p = {best_p:.3f}")
            print(f"  Maximum theoretical speedup: {1.0/(1.0-best_p):.1f}×")

            # Write fitted p to file
            with open("amdahl_fitted_p.txt", "w") as f:
                f.write(f"{best_p}\n")
    else:
        print("\nTo measure actual multi-GPU performance, run with --measure flag")
        print("  (requires GPU node with multiple GPUs available)")
