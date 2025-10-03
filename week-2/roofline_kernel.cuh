/*
 * Roofline Analysis Polynomial Kernel
 *
 * Computes b[i] = a[i] + a[i]^2 + ... + a[i]^k for roofline model demonstration.
 * Varies K to sweep arithmetic intensity from memory-bound to compute-bound.
 * See README.md for detailed explanation.
 */

#pragma once

/*
 * compute_k_terms: Polynomial evaluation kernel
 *
 * Parameters:
 *   a: Input array (const __restrict__ enables compiler optimizations)
 *   b: Output array (__restrict__ guarantees no aliasing)
 *   n: Number of elements
 *   k: Polynomial degree (controls arithmetic intensity: AI = k/4)
 *
 * extern "C" linkage: Required for CuPy RawModule interoperability
 */
extern "C" __global__ void compute_k_terms(const float* __restrict__ a,
                                            float* __restrict__ b,
                                            int n,
                                            int k) {
    // Calculate global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: ensure we don't access beyond array bounds
    if (i < n) {
        // Load input value from global memory (4-byte read)
        float x = a[i];

        // Initialize accumulator and running power of x
        float result = 0.0f;
        float power = x;  // Start with x^1

        // Compute polynomial sum: x + x^2 + ... + x^k
        // Loop performs K iterations:
        //   - Each iteration: 1 addition (result += power) + 1 multiplication (power *= x)
        //   - Total: 2K FLOPs
        // All operations use registers only - no memory traffic
        for (int j = 0; j < k; ++j) {
            result += power;    // Accumulate current power term
            power *= x;         // Advance to next power
        }

        // Write result to global memory (4-byte write)
        // Total memory traffic: 4 bytes (read) + 4 bytes (write) = 8 bytes
        b[i] = result;
    }
}

