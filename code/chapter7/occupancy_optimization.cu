// occupancy_optimization.cu - Occupancy and Resource Optimization Demo
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_occupancy.h>

// Kernel with different shared memory usage to demonstrate occupancy
template<int BLOCK_SIZE, int SHARED_MEM_SIZE>
__global__ void occupancyTestKernel(float *data, int n) {
    __shared__ float sharedMem[SHARED_MEM_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load to shared memory
    if (tid < SHARED_MEM_SIZE && idx < n) {
        sharedMem[tid] = data[idx];
    }
    __syncthreads();

    // Simple computation using shared memory
    if (idx < n) {
        float value = sharedMem[tid % SHARED_MEM_SIZE];
        data[idx] = sqrtf(value) + cosf(value);
    }
}

// Kernel with high register usage
__global__ void highRegisterKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Use many registers to simulate complex computation
        float r0 = data[idx];
        float r1 = r0 * 1.1f;
        float r2 = r1 * 1.2f;
        float r3 = r2 * 1.3f;
        float r4 = r3 * 1.4f;
        float r5 = r4 * 1.5f;
        float r6 = r5 * 1.6f;
        float r7 = r6 * 1.7f;
        float r8 = r7 * 1.8f;
        float r9 = r8 * 1.9f;
        float r10 = r9 * 2.0f;
        float r11 = r10 * 2.1f;
        float r12 = r11 * 2.2f;
        float r13 = r12 * 2.3f;
        float r14 = r13 * 2.4f;
        float r15 = r14 * 2.5f;

        data[idx] = r15;
    }
}

// Launch bounds example - hint to compiler about thread configuration
__global__ void __launch_bounds__(256, 4)
launchBoundsKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        data[idx] = sqrtf(data[idx]) + 1.0f;
    }
}

void calculateOccupancy() {
    printf("=== Occupancy Analysis ===\n\n");

    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Device: %s\n", prop.name);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Shared memory per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
    printf("Registers per block: %d\n", prop.regsPerBlock);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Warp size: %d\n\n", prop.warpSize);

    // Test different block sizes
    int blockSizes[] = {32, 64, 128, 256, 512, 1024};

    printf("=== Theoretical Occupancy for Different Block Sizes ===\n");
    for (int i = 0; i < 6; i++) {
        int blockSize = blockSizes[i];
        if (blockSize > prop.maxThreadsPerBlock) continue;

        int maxActiveBlocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks,
            occupancyTestKernel<256, 256>,
            blockSize,
            0
        );

        int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / prop.warpSize;
        int warpsPerBlock = blockSize / prop.warpSize;
        int activeWarps = maxActiveBlocks * warpsPerBlock;
        float occupancy = (float)activeWarps / maxWarpsPerSM * 100.0f;

        printf("Block size: %4d | Max active blocks/SM: %2d | Occupancy: %.1f%%\n",
               blockSize, maxActiveBlocks, occupancy);
    }
    printf("\n");
}

void testDynamicSharedMemory() {
    const int N = 1024 * 1024;
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    // Initialize data
    cudaMemset(d_data, 1, N * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("=== Dynamic Shared Memory Impact ===\n");

    // Test different shared memory sizes
    int sharedMemSizes[] = {0, 1024, 4096, 16384, 32768};
    const char* memLabels[] = {"0KB", "1KB", "4KB", "16KB", "32KB"};

    for (int i = 0; i < 5; i++) {
        int sharedMemSize = sharedMemSizes[i];
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;

        // Check max active blocks with this shared memory
        int maxActiveBlocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks,
            occupancyTestKernel<256, 1>,  // Use minimal template for calculation
            blockSize,
            sharedMemSize
        );

        if (maxActiveBlocks == 0) {
            printf("Shared mem: %s - Cannot launch (exceeds limit)\n", memLabels[i]);
            continue;
        }

        // Measure performance
        cudaEventRecord(start);
        for (int j = 0; j < 100; j++) {
            occupancyTestKernel<256, 1><<<gridSize, blockSize, sharedMemSize>>>(d_data, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time = 0;
        cudaEventElapsedTime(&time, start, stop);

        printf("Shared mem: %s | Max blocks/SM: %d | Time: %.2f ms\n",
               memLabels[i], maxActiveBlocks, time);
    }
    printf("\n");

    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void testOptimalLaunchConfig() {
    const int N = 10 * 1024 * 1024;
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    printf("=== Optimal Launch Configuration ===\n");

    // Method 1: Manual calculation
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    printf("Manual config: Grid=%d, Block=%d\n", gridSize, blockSize);

    // Method 2: Occupancy-based launch config
    int minGridSize, optBlockSize;
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &optBlockSize,
        launchBoundsKernel,
        0,  // No dynamic shared memory
        N   // Problem size
    );
    int optGridSize = (N + optBlockSize - 1) / optBlockSize;
    printf("Occupancy-based config: Grid=%d, Block=%d\n", optGridSize, optBlockSize);

    // Method 3: With shared memory constraint
    cudaOccupancyMaxPotentialBlockSizeVariableSMem(
        &minGridSize,
        &optBlockSize,
        occupancyTestKernel<256, 256>,
        [](int blockSize) -> size_t {
            return blockSize * sizeof(float);  // Dynamic shared memory function
        },
        0  // Block size limit (0 = no limit)
    );
    optGridSize = (N + optBlockSize - 1) / optBlockSize;
    printf("With shared memory: Grid=%d, Block=%d\n\n", optGridSize, optBlockSize);

    // Test performance with different configurations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Test manual config
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        launchBoundsKernel<<<gridSize, blockSize>>>(d_data, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float manualTime = 0;
    cudaEventElapsedTime(&manualTime, start, stop);

    // Test optimal config
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        launchBoundsKernel<<<optGridSize, optBlockSize>>>(d_data, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float optimalTime = 0;
    cudaEventElapsedTime(&optimalTime, start, stop);

    printf("Performance Comparison:\n");
    printf("Manual config: %.2f ms\n", manualTime);
    printf("Optimal config: %.2f ms\n", optimalTime);
    printf("Speedup: %.2fx\n", manualTime / optimalTime);

    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    calculateOccupancy();
    testDynamicSharedMemory();
    testOptimalLaunchConfig();

    return 0;
}