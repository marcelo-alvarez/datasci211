#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <cuda_runtime.h>

#include "roofline_kernel.cuh"

#define CUDA_CHECK(expr) do {                                  \
  cudaError_t _e = (expr);                                     \
  if (_e != cudaSuccess) {                                     \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
            __FILE__, __LINE__, cudaGetErrorString(_e));       \
    exit(1);                                                   \
  }                                                            \
} while (0)

struct SweepResult {
  int k;
  double ai;
  double mean_gflops;
  double mean_bw;
  double percent_peak;
};

SweepResult run_sweep(size_t N, int k, const float* da, float* db) {
  const int BLOCK = 256;
  const int GRID = (int)((N + BLOCK - 1) / BLOCK);

  // Arithmetic intensity calculation:
  // Per element: k additions + k multiplications = 2k FLOPs
  // Memory: 1 read (4 bytes) + 1 write (4 bytes) = 8 bytes
  double flops_per_element = 2.0 * k;
  double bytes_per_element = 8.0;
  double ai = flops_per_element / bytes_per_element;

  printf("\nK=%d (AI=%.3f flops/byte):\n", k, ai);

  // Warmup runs (silent)
  for (int w = 0; w < 3; ++w) {
    compute_k_terms<<<GRID, BLOCK>>>(da, db, N, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Timed runs
  const int NRUNS = 10;
  std::vector<float> times(NRUNS);
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  for (int run = 0; run < NRUNS; ++run) {
    cudaEventRecord(t0);
    compute_k_terms<<<GRID, BLOCK>>>(da, db, N, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&times[run], t0, t1);
  }

  // Calculate statistics
  float sum_ms = 0.0f;
  for (int i = 0; i < NRUNS; ++i) sum_ms += times[i];
  float mean_ms = sum_ms / NRUNS;

  // Calculate RMS for error checking
  float sum_sq_diff = 0.0f;
  for (int i = 0; i < NRUNS; ++i) {
    float diff = times[i] - mean_ms;
    sum_sq_diff += diff * diff;
  }
  float rms = sqrt(sum_sq_diff / NRUNS);
  float rms_percent = (rms / mean_ms) * 100.0;

  double mean_gflops = (N * flops_per_element / 1e9) / (mean_ms / 1e3);
  double mean_bw = (N * bytes_per_element / 1e9) / (mean_ms / 1e3);
  double percent_peak = 100.0 * mean_gflops / 67000.0;

  printf("  Mean: %.3f ms (%.2f GFLOPS, %.2f GB/s, %.2f%% peak)",
         mean_ms, mean_gflops, mean_bw, percent_peak);

  const float RMS_TOLERANCE = 5.0;  // 5% tolerance
  if (rms_percent > RMS_TOLERANCE) {
    printf(" [WARNING: RMS=%.2f%% > %.1f%%]", rms_percent, RMS_TOLERANCE);
  }
  printf("\n");

  cudaEventDestroy(t0);
  cudaEventDestroy(t1);

  return {k, ai, mean_gflops, mean_bw, percent_peak};
}

int main() {
  // GPU info
  int dev = 0;
  CUDA_CHECK(cudaSetDevice(dev));
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  printf("Using GPU 0: %s | CC %d.%d | SMs=%d | globalMem=%.1f GiB\n",
         p.name, p.major, p.minor, p.multiProcessorCount,
         p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  printf("FP32 Peak: 67 TFLOPS | HBM3 Bandwidth: 3.0 TB/s\n");

  // Problem size: 2^26 elements (256 MB per array)
  const size_t N = 1 << 26;  // 67M elements
  const size_t BYTES = N * sizeof(float);

  // Allocate and initialize
  std::vector<float> ha(N, 1.5f);
  float *da = nullptr, *db = nullptr;
  CUDA_CHECK(cudaMalloc(&da, BYTES));
  CUDA_CHECK(cudaMalloc(&db, BYTES));
  CUDA_CHECK(cudaMemcpy(da, ha.data(), BYTES, cudaMemcpyHostToDevice));

  // Sweep through K values (arithmetic intensity)
  int k_values[] = {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000};

  FILE* fp = fopen("roofline_cuda.csv", "w");
  fprintf(fp, "k,ai,gflops,bandwidth,percent_peak\n");

  for (int k : k_values) {
    auto r = run_sweep(N, k, da, db);
    fprintf(fp, "%d,%.6f,%.2f,%.2f,%.2f\n",
            r.k, r.ai, r.mean_gflops, r.mean_bw, r.percent_peak);
  }

  fclose(fp);

  cudaFree(da);
  cudaFree(db);
  return 0;
}
