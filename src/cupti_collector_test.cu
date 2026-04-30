/**
 * cupti_collector_test.cu — smoke test for cupti_collector
 * Initializes collector, runs a trivial kernel, prints kind status and phase summary.
 * Build: /usr/local/cuda-13.0/bin/nvcc -O2 -std=c++17 -I./include
 *        src/cupti_collector.cu src/cupti_collector_test.cu
 *        -o cupti_collector_test -lcudart -lcupti
 */

#include "cupti_collector.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void dummy_kernel(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = a[i] * 2.0f;
}

int main(void) {
    CuptiCollector col;

    printf("=== cupti_collector smoke test ===\n");

    if (cupti_collector_init(&col) != 0) {
        fprintf(stderr, "init failed\n");
        return 1;
    }

    /* Phase 1 — synthetic kernel */
    cupti_collector_phase_begin(&col, "synthetic_kernel");

    float* d = NULL;
    cudaMalloc(&d, 1024 * sizeof(float));
    dummy_kernel<<<4, 256>>>(d, 1024);
    cudaDeviceSynchronize();
    cudaFree(d);

    cupti_collector_phase_end(&col);

    /* Phase 2 — memcpy */
    cupti_collector_phase_begin(&col, "memcpy_test");

    float* h = (float*)malloc(1024 * sizeof(float));
    float* d2 = NULL;
    cudaMalloc(&d2, 1024 * sizeof(float));
    cudaMemcpy(d2, h, 1024 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(h, d2, 1024 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d2);
    free(h);

    cupti_collector_phase_end(&col);

    cupti_collector_print(&col);
    cupti_collector_write_json(&col, "cupti_collector_test.json");
    printf("\nJSON written: cupti_collector_test.json\n");

    cupti_collector_destroy(&col);
    return 0;
}
