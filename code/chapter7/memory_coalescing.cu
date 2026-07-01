// memory_coalescing.cu - Memory Coalescing Optimization Demo
#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

#define N 1024 * 1024
#define BLOCK_SIZE 256

// Uncoalesced memory access (stride access)
__global__ void uncoalescedAccess(float *input, float *output, int stride) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < N / stride) {
        output[gid] = input[gid * stride] * 2.0f;
    }
}

// Coalesced memory access (sequential access)
__global__ void coalescedAccess(float *input, float *output) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < N) {
        output[gid] = input[gid] * 2.0f;
    }
}

// Structure of Arrays (SoA) - Better for GPU
struct SoA {
    float *x;
    float *y;
    float *z;
};

// Array of Structures (AoS) - Worse for GPU
struct AoS {
    float x;
    float y;
    float z;
};

__global__ void processAoS(AoS *data, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Non-coalesced: accessing x from different structures
        result[idx] = data[idx].x + data[idx].y + data[idx].z;
    }
}

__global__ void processSoA(float *x, float *y, float *z, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Coalesced: all threads access consecutive elements
        result[idx] = x[idx] + y[idx] + z[idx];
    }
}

// Bank conflict example
__global__ void bankConflictDemo(float *output) {
    __shared__ float sharedMem[32][33];  // 33 to avoid bank conflicts (padding)

    int tid = threadIdx.x;

    // Write with potential bank conflicts
    sharedMem[tid % 32][tid / 32] = tid * 1.0f;
    __syncthreads();

    // Read with reduced bank conflicts due to padding
    output[tid] = sharedMem[tid / 32][tid % 32];
}

void testMemoryCoalescing() {
    float *d_input, *d_output;
    const size_t bytes = N * sizeof(float);

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // Initialize input
    float *h_input = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i;
    }
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Test coalesced access
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        coalescedAccess<<<blocks, BLOCK_SIZE>>>(d_input, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float coalescedTime = 0;
    cudaEventElapsedTime(&coalescedTime, start, stop);

    // Test uncoalesced access (stride = 32)
    int stride = 32;
    blocks = (N/stride + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        uncoalescedAccess<<<blocks, BLOCK_SIZE>>>(d_input, d_output, stride);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float uncoalescedTime = 0;
    cudaEventElapsedTime(&uncoalescedTime, start, stop);

    printf("=== Memory Coalescing Test ===\n");
    printf("Coalesced access: %.3f ms\n", coalescedTime);
    printf("Uncoalesced access (stride=%d): %.3f ms\n", stride, uncoalescedTime);
    printf("Speedup from coalescing: %.2fx\n\n", uncoalescedTime / coalescedTime);

    free(h_input);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void testSoAvsAoS() {
    const int n = 1024 * 1024;
    const size_t bytesFloat = n * sizeof(float);
    const size_t bytesAoS = n * sizeof(AoS);

    // Allocate SoA
    float *d_x, *d_y, *d_z, *d_resultSoA;
    cudaMalloc(&d_x, bytesFloat);
    cudaMalloc(&d_y, bytesFloat);
    cudaMalloc(&d_z, bytesFloat);
    cudaMalloc(&d_resultSoA, bytesFloat);

    // Allocate AoS
    AoS *d_aos;
    float *d_resultAoS;
    cudaMalloc(&d_aos, bytesAoS);
    cudaMalloc(&d_resultAoS, bytesFloat);

    // Initialize data
    float *h_x = (float*)malloc(bytesFloat);
    float *h_y = (float*)malloc(bytesFloat);
    float *h_z = (float*)malloc(bytesFloat);
    AoS *h_aos = (AoS*)malloc(bytesAoS);

    for (int i = 0; i < n; i++) {
        h_x[i] = i * 1.0f;
        h_y[i] = i * 2.0f;
        h_z[i] = i * 3.0f;
        h_aos[i].x = i * 1.0f;
        h_aos[i].y = i * 2.0f;
        h_aos[i].z = i * 3.0f;
    }

    cudaMemcpy(d_x, h_x, bytesFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytesFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, bytesFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_aos, h_aos, bytesAoS, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Test SoA
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        processSoA<<<blocks, BLOCK_SIZE>>>(d_x, d_y, d_z, d_resultSoA, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float soaTime = 0;
    cudaEventElapsedTime(&soaTime, start, stop);

    // Test AoS
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        processAoS<<<blocks, BLOCK_SIZE>>>(d_aos, d_resultAoS, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float aosTime = 0;
    cudaEventElapsedTime(&aosTime, start, stop);

    printf("=== SoA vs AoS Test ===\n");
    printf("Structure of Arrays (SoA): %.3f ms\n", soaTime);
    printf("Array of Structures (AoS): %.3f ms\n", aosTime);
    printf("SoA Speedup: %.2fx\n\n", aosTime / soaTime);

    // Cleanup
    free(h_x);
    free(h_y);
    free(h_z);
    free(h_aos);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_aos);
    cudaFree(d_resultSoA);
    cudaFree(d_resultAoS);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    testMemoryCoalescing();
    testSoAvsAoS();

    return 0;
}