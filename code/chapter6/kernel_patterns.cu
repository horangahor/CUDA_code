// kernel_patterns.cu - Common CUDA Kernel Patterns
#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// Pattern 1: Map (Element-wise operation)
__global__ void mapKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = sqrtf(input[idx]) + 1.0f;
    }
}

// Pattern 2: Reduce (Sum reduction)
__global__ void reduceKernel(float *input, float *output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load to shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Pattern 3: Stencil (1D convolution)
__global__ void stencilKernel(float *input, float *output, int n, int radius) {
    extern __shared__ float temp[];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;

    // Load with halo
    int sharedIdx = lid + radius;
    if (gid < n) {
        temp[sharedIdx] = input[gid];
    }

    // Load halo elements
    if (lid < radius) {
        if (gid >= radius) {
            temp[lid] = input[gid - radius];
        }
        if (gid + blockDim.x < n) {
            temp[sharedIdx + blockDim.x] = input[gid + blockDim.x];
        }
    }
    __syncthreads();

    // Apply stencil (simple 3-point averaging)
    if (gid > 0 && gid < n - 1) {
        output[gid] = (temp[sharedIdx - 1] + temp[sharedIdx] + temp[sharedIdx + 1]) / 3.0f;
    }
}

// Pattern 4: Scatter/Gather (Histogram)
__global__ void histogramKernel(int *input, int *histogram, int n, int numBins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int bin = input[idx] % numBins;
        atomicAdd(&histogram[bin], 1);
    }
}

// Helper function to print arrays
void printArray(const char* name, float* arr, int n) {
    printf("%s: ", name);
    for (int i = 0; i < n && i < 10; i++) {
        printf("%.2f ", arr[i]);
    }
    if (n > 10) printf("...");
    printf("\n");
}

int main() {
    const int N = 1024;
    const int bytes = N * sizeof(float);

    // Allocate memory
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    float *d_input, *d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // Initialize input
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i % 100);
    }

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Test Map Pattern
    printf("=== Map Pattern (sqrt + 1) ===\n");
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mapKernel<<<blocks, BLOCK_SIZE>>>(d_input, d_output, N);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    printArray("Map result", h_output, N);

    // Test Reduce Pattern
    printf("\n=== Reduce Pattern (Sum) ===\n");
    float *d_partial;
    cudaMalloc(&d_partial, blocks * sizeof(float));
    reduceKernel<<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_partial, N);

    float *h_partial = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float totalSum = 0;
    for (int i = 0; i < blocks; i++) {
        totalSum += h_partial[i];
    }
    printf("Total sum: %.2f\n", totalSum);

    // Test Stencil Pattern
    printf("\n=== Stencil Pattern (3-point average) ===\n");
    int radius = 1;
    int sharedMemSize = (BLOCK_SIZE + 2 * radius) * sizeof(float);
    stencilKernel<<<blocks, BLOCK_SIZE, sharedMemSize>>>(d_input, d_output, N, radius);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    printArray("Stencil result", h_output, N);

    // Test Histogram Pattern
    printf("\n=== Histogram Pattern ===\n");
    const int numBins = 10;
    int *h_intInput = (int*)malloc(N * sizeof(int));
    int *d_intInput, *d_histogram;
    int *h_histogram = (int*)calloc(numBins, sizeof(int));

    for (int i = 0; i < N; i++) {
        h_intInput[i] = i % 100;
    }

    cudaMalloc(&d_intInput, N * sizeof(int));
    cudaMalloc(&d_histogram, numBins * sizeof(int));
    cudaMemset(d_histogram, 0, numBins * sizeof(int));

    cudaMemcpy(d_intInput, h_intInput, N * sizeof(int), cudaMemcpyHostToDevice);
    histogramKernel<<<blocks, BLOCK_SIZE>>>(d_intInput, d_histogram, N, numBins);
    cudaMemcpy(h_histogram, d_histogram, numBins * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Histogram bins: ");
    for (int i = 0; i < numBins; i++) {
        printf("[%d]=%d ", i, h_histogram[i]);
    }
    printf("\n");

    // Cleanup
    free(h_input);
    free(h_output);
    free(h_partial);
    free(h_intInput);
    free(h_histogram);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_partial);
    cudaFree(d_intInput);
    cudaFree(d_histogram);

    return 0;
}