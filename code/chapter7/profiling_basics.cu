// profiling_basics.cu - Basic Profiling and Performance Measurement
#include <stdio.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#define N 1024 * 1024
#define BLOCK_SIZE 256

// Simple kernel for profiling
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// More complex kernel with shared memory
__global__ void reductionKernel(float *input, float *output, int n) {
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

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Function to measure kernel execution time using events
void measureWithEvents() {
    printf("=== Timing with CUDA Events ===\n");

    float *d_a, *d_b, *d_c;
    const size_t bytes = N * sizeof(float);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Initialize data
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Create events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Warm-up
    vectorAdd<<<blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    // Measure kernel execution
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        vectorAdd<<<blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel execution time (100 iterations): %.3f ms\n", milliseconds);
    printf("Average time per kernel: %.3f ms\n", milliseconds / 100.0f);

    // Calculate bandwidth
    size_t totalBytes = 3 * bytes * 100;  // 2 reads + 1 write, 100 iterations
    float bandwidth = (totalBytes / 1e9) / (milliseconds / 1000.0f);
    printf("Effective bandwidth: %.2f GB/s\n\n", bandwidth);

    // Cleanup
    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Function to demonstrate NVTX markers for profiling
void measureWithNVTX() {
    printf("=== Using NVTX Markers (visible in Nsight Systems) ===\n");

    float *d_data, *d_output;
    const size_t bytes = N * sizeof(float);

    // NVTX range for initialization
    nvtxRangePush("Initialization");
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_output, bytes / BLOCK_SIZE);

    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 1.0f;
    }
    nvtxRangePop();

    // NVTX range for memory transfer
    nvtxRangePush("H2D Transfer");
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    nvtxRangePop();

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // NVTX range for kernel execution
    nvtxRangePush("Reduction Kernel");
    reductionKernel<<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_data, d_output, N);
    cudaDeviceSynchronize();
    nvtxRangePop();

    // NVTX range for result transfer
    nvtxRangePush("D2H Transfer");
    float *h_output = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(h_output, d_output, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    nvtxRangePop();

    // Calculate final sum
    nvtxRangePush("CPU Reduction");
    float totalSum = 0;
    for (int i = 0; i < blocks; i++) {
        totalSum += h_output[i];
    }
    nvtxRangePop();

    printf("Reduction result: %.0f\n");
    printf("(Run with nsys profile ./profiling_basics to see NVTX markers)\n\n");

    // Cleanup
    free(h_data);
    free(h_output);
    cudaFree(d_data);
    cudaFree(d_output);
}

// Function to get and display runtime statistics
void displayMetrics() {
    printf("=== Runtime Metrics ===\n");

    // Get device properties
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("Device %d: %s\n", device, prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Clock Rate: %.2f GHz\n", prop.clockRate / 1e6);
    printf("Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);

    // Calculate theoretical bandwidth
    float theoreticalBW = (prop.memoryClockRate * 1000.0f * 2 * prop.memoryBusWidth) / 8.0f / 1e9;
    printf("Theoretical Bandwidth: %.2f GB/s\n", theoreticalBW);

    // Calculate theoretical compute
    float theoreticalCompute = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor * prop.clockRate / 1e6;
    printf("Theoretical SP GFLOPS: %.2f\n\n", theoreticalCompute);
}

// Function to compare different memory access patterns
void compareMemoryPatterns() {
    printf("=== Memory Access Pattern Comparison ===\n");

    const int size = 1024 * 1024 * 10;
    float *d_input, *d_output;

    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Test 1: Sequential access (coalesced)
    cudaEventRecord(start);
    vectorAdd<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_input, d_input, d_output, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float sequentialTime = 0;
    cudaEventElapsedTime(&sequentialTime, start, stop);

    printf("Sequential access time: %.3f ms\n", sequentialTime);

    // Calculate effective bandwidth
    size_t bytes = 3 * size * sizeof(float);  // 2 reads + 1 write
    float bandwidth = (bytes / 1e9) / (sequentialTime / 1000.0f);
    printf("Effective bandwidth: %.2f GB/s\n\n", bandwidth);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    displayMetrics();
    measureWithEvents();
    measureWithNVTX();
    compareMemoryPatterns();

    printf("=== Profiling Tips ===\n");
    printf("1. Use 'nvprof ./profiling_basics' for quick profiling\n");
    printf("2. Use 'nsys profile ./profiling_basics' for timeline analysis\n");
    printf("3. Use 'ncu ./profiling_basics' for detailed kernel analysis\n");
    printf("4. Add --metrics all to nvprof for detailed metrics\n");

    return 0;
}