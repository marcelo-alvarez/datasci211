#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr) do {                                  \
  cudaError_t _e = (expr);                                     \
  if (_e != cudaSuccess) {                                     \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
            __FILE__, __LINE__, cudaGetErrorString(_e));       \
    exit(1);                                                   \
  }                                                            \
} while (0)

__global__ void vadd(const float* __restrict__ a,
                     const float* __restrict__ b,
                     float* __restrict__ c,
                     size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  // --- Report GPU info ---
  int dev = 0, ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  if (ndev == 0) { fprintf(stderr, "No CUDA device visible.\n"); return 1; }
  CUDA_CHECK(cudaSetDevice(dev));
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  printf("Using GPU 0: %s | CC %d.%d | SMs=%d | globalMem=%.1f GiB\n",
         p.name, p.major, p.minor, p.multiProcessorCount,
         p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

  // --- Problem size (keep small for quick demo) ---
  const size_t N = 1u << 22;           // ~4,194,304 elements
  const size_t BYTES = N * sizeof(float);
  printf("Vector length N=%zu (%.2f MiB per vector)\n",
         N, BYTES / (1024.0 * 1024.0));

  // --- Host init ---
  std::vector<float> ha(N), hb(N), hc(N);
  for (size_t i = 0; i < N; ++i) {
    ha[i] = 1.0f; hb[i] = 2.0f;
  }

  // --- Device alloc ---
  float *da=nullptr, *db=nullptr, *dc=nullptr;
  CUDA_CHECK(cudaMalloc(&da, BYTES));
  CUDA_CHECK(cudaMalloc(&db, BYTES));
  CUDA_CHECK(cudaMalloc(&dc, BYTES));

  // --- Copy H->D ---
  CUDA_CHECK(cudaMemcpy(da, ha.data(), BYTES, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb.data(), BYTES, cudaMemcpyHostToDevice));

  // --- Launch ---
  const int BLOCK = 256;
  const int GRID  = (int)((N + BLOCK - 1) / BLOCK);

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);

  vadd<<<GRID, BLOCK>>>(da, db, dc, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, t0, t1);

  // --- Copy D->H and check ---
  CUDA_CHECK(cudaMemcpy(hc.data(), dc, BYTES, cudaMemcpyDeviceToHost));
  double checksum = 0.0;
  for (size_t i = 0; i < N; ++i) checksum += hc[i];
  printf("Checksum=%.0f (expected %.0f)\n", checksum, 3.0 * (double)N);

  // --- Simple perf estimate ---
  double gbytes = 3.0 * BYTES / 1e9;           // a+b->c: 3 arrays touched
  double secs = ms / 1e3;
  printf("Kernel time=%.3f ms, approx bandwidth=%.2f GB/s\n",
         ms, gbytes / secs);

  // --- Cleanup ---
  cudaFree(da); cudaFree(db); cudaFree(dc);
  cudaEventDestroy(t0); cudaEventDestroy(t1);
  printf("Hello GPU (CUDA) completed.\n");
  return 0;
}
