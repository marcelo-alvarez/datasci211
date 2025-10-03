/*
 * Roofline Model Performance Analysis: CUDA Implementation
 *
 * Measures GPU performance across varying arithmetic intensities by sweeping
 * polynomial degree K from 1 to 1000. Shares kernel with CuPy version for
 * fair performance comparison. See README.md for detailed explanation.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <cuda_runtime.h>

#include "roofline_kernel.cuh"

/*
 * CUDA_CHECK: Error checking macro for CUDA API calls
 *
 * CUDA API functions return cudaError_t status codes. This macro:
 * - Checks if the operation succeeded
 * - Prints file/line information for debugging
 * - Terminates program on error (fail-fast approach)
 *
 * Usage: CUDA_CHECK(cudaMalloc(&ptr, size));
 */
#define CUDA_CHECK(expr) do {                                  \
  cudaError_t _e = (expr);                                     \
  if (_e != cudaSuccess) {                                     \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
            __FILE__, __LINE__, cudaGetErrorString(_e));       \
    exit(1);                                                   \
  }                                                            \
} while (0)

/*
 * SweepResult: Stores performance metrics for one K value
 *
 * Fields:
 *   k: Polynomial degree (determines arithmetic intensity)
 *   ai: Arithmetic intensity in FLOPs/byte (= 2k / 8 = k/4)
 *   mean_gflops: Achieved performance in GFLOPS (billions of FLOPs per second)
 *   mean_bw: Effective memory bandwidth in GB/s
 *   percent_peak: Percentage of theoretical peak FP32 performance (67 TFLOPS)
 */
struct SweepResult {
  int k;
  double ai;
  double mean_gflops;
  double mean_bw;
  double percent_peak;
};

/*
 * run_sweep: Measure kernel performance for a specific K value
 *
 * Performs timing measurements for the polynomial kernel with a given
 * degree K. This function implements best practices for GPU benchmarking:
 *
 * 1. WARMUP RUNS: First few kernel launches may be slower due to:
 *    - JIT compilation overhead
 *    - GPU frequency scaling (GPU may be in low-power state)
 *    - Cache effects
 *    We run 3 warmup iterations and discard these timings.
 *
 * 2. MULTIPLE TIMED RUNS: Run kernel 10 times and compute statistics.
 *    This helps identify:
 *    - Mean performance (primary metric)
 *    - Timing variance (RMS) to detect interference from system noise
 *
 * 3. EVENT-BASED TIMING: Use cudaEvent for GPU-side timing rather than
 *    CPU-side clock. This eliminates:
 *    - CPU-GPU synchronization overhead
 *    - CPU timing precision limitations
 *    - Asynchronous kernel launch effects
 *
 * Parameters:
 *   N: Number of array elements
 *   k: Polynomial degree (controls arithmetic intensity)
 *   da: Device input array pointer
 *   db: Device output array pointer
 *
 * Returns:
 *   SweepResult containing performance metrics and AI
 */
SweepResult run_sweep(size_t N, int k, const float* da, float* db) {
  const int BLOCK = 256;  // Threads per block (typical choice for good occupancy)
  const int GRID = (int)((N + BLOCK - 1) / BLOCK);  // Ceiling division for grid size

  // ARITHMETIC INTENSITY CALCULATION:
  // Per element: k additions + k multiplications = 2k FLOPs
  // Memory access: 1 read (4 bytes) + 1 write (4 bytes) = 8 bytes
  // AI = FLOPs / Bytes = 2k / 8 = k/4 FLOPs/byte
  double flops_per_element = 2.0 * k;
  double bytes_per_element = 8.0;
  double ai = flops_per_element / bytes_per_element;

  printf("\nK=%d (AI=%.3f flops/byte):\n", k, ai);

  // WARMUP PHASE: Run kernel 3 times to eliminate cold-start effects
  // GPUs may need initial launches to:
  // - Reach full clock frequency (power management)
  // - Complete JIT compilation
  // - Prime memory caches
  for (int w = 0; w < 3; ++w) {
    compute_k_terms<<<GRID, BLOCK>>>(da, db, N, k);
    CUDA_CHECK(cudaGetLastError());  // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize());  // Wait for kernel completion
  }

  // TIMED RUNS PHASE: Measure actual performance over multiple iterations
  const int NRUNS = 10;  // Number of timing measurements to collect
  std::vector<float> times(NRUNS);

  // Create CUDA events for GPU-side timing
  // Events are inserted into the GPU command stream and allow precise timing
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  for (int run = 0; run < NRUNS; ++run) {
    // Record start event into the default stream
    cudaEventRecord(t0);

    // Launch kernel
    compute_k_terms<<<GRID, BLOCK>>>(da, db, N, k);
    CUDA_CHECK(cudaGetLastError());  // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize());  // Ensure kernel completes

    // Record stop event
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);  // Wait for stop event to complete

    // Compute elapsed time in milliseconds between events
    cudaEventElapsedTime(&times[run], t0, t1);
  }

  // STATISTICS CALCULATION:
  // Compute mean execution time
  float sum_ms = 0.0f;
  for (int i = 0; i < NRUNS; ++i) sum_ms += times[i];
  float mean_ms = sum_ms / NRUNS;

  // Calculate RMS deviation to assess timing stability
  float sum_sq_diff = 0.0f;
  for (int i = 0; i < NRUNS; ++i) {
    float diff = times[i] - mean_ms;
    sum_sq_diff += diff * diff;
  }
  float rms = sqrt(sum_sq_diff / NRUNS);
  float rms_percent = (rms / mean_ms) * 100.0;

  // PERFORMANCE METRICS:
  // GFLOPS = (Total FLOPs / 10^9) / (Time in seconds)
  double mean_gflops = (N * flops_per_element / 1e9) / (mean_ms / 1e3);

  // Effective bandwidth = (Total bytes / 10^9) / (Time in seconds)
  double mean_bw = (N * bytes_per_element / 1e9) / (mean_ms / 1e3);

  // Percentage of H100's theoretical peak FP32 performance (67 TFLOPS)
  double percent_peak = 100.0 * mean_gflops / 67000.0;

  printf("  Mean: %.3f ms (%.2f GFLOPS, %.2f GB/s, %.2f%% peak)",
         mean_ms, mean_gflops, mean_bw, percent_peak);

  // Warn if timing variance is too high (>5%)
  // High variance suggests unreliable measurements
  const float RMS_TOLERANCE = 5.0;  // 5% tolerance
  if (rms_percent > RMS_TOLERANCE) {
    printf(" [WARNING: RMS=%.2f%% > %.1f%%]", rms_percent, RMS_TOLERANCE);
  }
  printf("\n");

  // Clean up event objects
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);

  return {k, ai, mean_gflops, mean_bw, percent_peak};
}

/*
 * main: Execute roofline analysis sweep
 *
 * WORKFLOW:
 * 1. Query GPU properties and display specifications
 * 2. Allocate device memory for input/output arrays
 * 3. Sweep through K values (varying arithmetic intensity)
 * 4. Write results to CSV for plotting
 */
int main() {
  // STEP 1: Query and display GPU information
  int dev = 0;
  CUDA_CHECK(cudaSetDevice(dev));  // Select first GPU
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

  // Display GPU specifications
  printf("Using GPU 0: %s | CC %d.%d | SMs=%d | globalMem=%.1f GiB\n",
         p.name, p.major, p.minor, p.multiProcessorCount,
         p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  printf("FP32 Peak: 67 TFLOPS | HBM3 Bandwidth: 3.0 TB/s\n");

  // STEP 2: Allocate memory
  // Problem size: 2^26 elements = 67M elements
  // Array size: 67M * 4 bytes/float = 268 MB per array
  // Large enough to avoid cache effects, small enough for quick iteration
  const size_t N = 1 << 26;  // 67M elements
  const size_t BYTES = N * sizeof(float);

  // Create host array with initial value 1.01 (avoids overflow for K<1000)
  // (value chosen to avoid special cases like 0, 1, or powers of 2)
  std::vector<float> ha(N, 1.01f);

  // Allocate device memory
  float *da = nullptr, *db = nullptr;
  CUDA_CHECK(cudaMalloc(&da, BYTES));  // Input array
  CUDA_CHECK(cudaMalloc(&db, BYTES));  // Output array

  // Copy input data to device
  CUDA_CHECK(cudaMemcpy(da, ha.data(), BYTES, cudaMemcpyHostToDevice));

  // STEP 3: Sweep through K values to vary arithmetic intensity
  // K range: 1 to 1000, chosen to span memory-bound to compute-bound regions
  // - K=1:    AI=0.25 FLOPs/byte  (memory-bound)
  // - K=50:   AI=12.5 FLOPs/byte  (near ridge point)
  // - K=1000: AI=250 FLOPs/byte   (deeply compute-bound)
  int k_values[] = {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000};

  // Open CSV file for output
  FILE* fp = fopen("roofline_cuda.csv", "w");
  fprintf(fp, "k,ai,gflops,bandwidth,percent_peak\n");

  // Run performance measurements for each K value
  for (int k : k_values) {
    auto r = run_sweep(N, k, da, db);

    // Write results to CSV
    // Format: k, arithmetic_intensity, GFLOPS, bandwidth, percent_of_peak
    fprintf(fp, "%d,%.6f,%.2f,%.2f,%.2f\n",
            r.k, r.ai, r.mean_gflops, r.mean_bw, r.percent_peak);
  }

  fclose(fp);

  // STEP 4: Clean up
  cudaFree(da);
  cudaFree(db);
  return 0;
}
