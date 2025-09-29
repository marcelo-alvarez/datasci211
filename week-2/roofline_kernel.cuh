#pragma once

extern "C" __global__ void compute_k_terms(const float* __restrict__ a,
                                            float* __restrict__ b,
                                            int n,
                                            int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        float result = 0.0f;
        float power = x;

        // Compute polynomial: x + x^2 + ... + x^k
        for (int j = 0; j < k; ++j) {
            result += power;
            power *= x;
        }

        b[i] = result;
    }
}

