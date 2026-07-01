#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>

// Error checking macro
#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                __FILE__, __LINE__, \
                static_cast<int>(error), \
                cudaGetErrorName(error), \
                cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// =======================================================
// Thread Hierarchy Demonstration
// =======================================================

__global__ void threadHierarchyKernel(int* output) {
    // Calculate global thread ID
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    int global_z = blockIdx.z * blockDim.z + threadIdx.z;

    // Calculate total threads per block and grid
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    int blocks_per_grid = gridDim.x * gridDim.y * gridDim.z;

    // Calculate linear indices
    int thread_id_in_block = threadIdx.z * (blockDim.x * blockDim.y) +
                            threadIdx.y * blockDim.x +
                            threadIdx.x;

    int block_id_in_grid = blockIdx.z * (gridDim.x * gridDim.y) +
                          blockIdx.y * gridDim.x +
                          blockIdx.x;

    int global_thread_id = block_id_in_grid * threads_per_block + thread_id_in_block;

    // Store information in output array
    if (global_thread_id < 1000) {  // Limit output size
        int base = global_thread_id * 10;
        output[base + 0] = threadIdx.x;
        output[base + 1] = threadIdx.y;
        output[base + 2] = threadIdx.z;
        output[base + 3] = blockIdx.x;
        output[base + 4] = blockIdx.y;
        output[base + 5] = blockIdx.z;
        output[base + 6] = thread_id_in_block;
        output[base + 7] = block_id_in_grid;
        output[base + 8] = global_thread_id;
        output[base + 9] = threads_per_block;
    }
}

// =======================================================
// Warp-Level Operations
// =======================================================

__global__ void warpLevelKernel(int* input, int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    int value = input[idx];

    #if __CUDA_ARCH__ >= 300
    // Warp-level sum using shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xFFFFFFFF, value, offset);
    }

    // Warp vote operations
    int all_positive = __all_sync(0xFFFFFFFF, input[idx] > 0);
    int any_negative = __any_sync(0xFFFFFFFF, input[idx] < 0);
    unsigned ballot = __ballot_sync(0xFFFFFFFF, input[idx] > 50);

    // Store results
    if (lane_id == 0) {
        output[idx / 32 * 4 + 0] = value;  // Warp sum
        output[idx / 32 * 4 + 1] = all_positive;
        output[idx / 32 * 4 + 2] = any_negative;
        output[idx / 32 * 4 + 3] = __popc(ballot);  // Count of values > 50
    }
    #else
    // Fallback for older architectures
    output[idx] = value;
    #endif
}

// =======================================================
// Block-Level Cooperative Operations
// =======================================================

__global__ void blockCooperativeKernel(float* input, float* output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (idx < n) {
        sdata[tid] = input[idx];
    } else {
        sdata[tid] = 0.0f;
    }

    __syncthreads();

    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Block-level broadcast
    float block_sum = sdata[0];
    __syncthreads();

    // Each thread computes normalized value
    if (idx < n) {
        float normalized = (block_sum > 0) ? input[idx] / block_sum : 0.0f;
        output[idx] = normalized;
    }
}

// =======================================================
// Grid-Level Coordination
// =======================================================

__device__ int global_counter = 0;

__global__ void gridCoordinationKernel(float* data, float* results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Phase 1: Each thread processes its data
    if (idx < n) {
        data[idx] = sinf(static_cast<float>(idx) * 0.001f);
    }

    __syncthreads();

    // Phase 2: Block coordination
    __shared__ float block_max;
    if (threadIdx.x == 0) {
        block_max = -1.0f;
    }
    __syncthreads();

    // Find block maximum
    if (idx < n) {
        atomicMax(&block_max, data[idx]);
    }
    __syncthreads();

    // Phase 3: Grid-level coordination
    if (threadIdx.x == 0) {
        int block_id = atomicAdd(&global_counter, 1);
        results[block_id] = block_max;
    }
}

// =======================================================
// Hardware Mapping Demonstration
// =======================================================

__global__ void hardwareMappingKernel(int* sm_info, int* warp_info) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;

    // Get SM ID (if available)
    #if __CUDA_ARCH__ >= 200
    unsigned int sm_id = __smid();
    #else
    unsigned int sm_id = 0;
    #endif

    // Store SM mapping information
    if (tid == 0) {
        sm_info[bid * 3 + 0] = sm_id;
        sm_info[bid * 3 + 1] = blockDim.x;
        sm_info[bid * 3 + 2] = gridDim.x;
    }

    // Store warp information
    if (lane_id == 0) {
        int warp_idx = bid * (blockDim.x / 32) + warp_id;
        if (warp_idx < 1000) {  // Limit output
            warp_info[warp_idx * 4 + 0] = sm_id;
            warp_info[warp_idx * 4 + 1] = bid;
            warp_info[warp_idx * 4 + 2] = warp_id;
            warp_info[warp_idx * 4 + 3] = 32;  // warp size
        }
    }
}

// =======================================================
// Launch Configuration Optimization
// =======================================================

class LaunchOptimizer {
public:
    static void demonstrateOccupancyOptimization() {
        printf("=== Occupancy Optimization ===\n");

        int device;
        CHECK_CUDA(cudaGetDevice(&device));

        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

        printf("Device: %s\n", prop.name);
        printf("SMs: %d\n", prop.multiProcessorCount);
        printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);

        // Test different block sizes
        int block_sizes[] = {32, 64, 128, 256, 512, 1024};

        for (int block_size : block_sizes) {
            int min_grid_size, opt_block_size;
            CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size, &opt_block_size, warpLevelKernel, 0, 0));

            int max_blocks_per_sm;
            CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_per_sm, warpLevelKernel, block_size, 0));

            float occupancy = (max_blocks_per_sm * block_size) /
                             static_cast<float>(prop.maxThreadsPerMultiProcessor);

            printf("Block size %4d: %d blocks/SM, %.1f%% occupancy\n",
                   block_size, max_blocks_per_sm, occupancy * 100);
        }

        printf("\n");
    }

    static dim3 calculateOptimalGrid(int total_elements, dim3 block_size) {
        dim3 grid_size;
        grid_size.x = (total_elements + block_size.x - 1) / block_size.x;
        grid_size.y = 1;
        grid_size.z = 1;
        return grid_size;
    }

    static dim3 calculate2DOptimalGrid(int width, int height, dim3 block_size) {
        dim3 grid_size;
        grid_size.x = (width + block_size.x - 1) / block_size.x;
        grid_size.y = (height + block_size.y - 1) / block_size.y;
        grid_size.z = 1;
        return grid_size;
    }
};

// =======================================================
// Test Functions
// =======================================================

void testThreadHierarchy() {
    printf("=== Thread Hierarchy Test ===\n");

    const int output_size = 1000 * 10;
    std::vector<int> h_output(output_size, -1);

    int* d_output;
    CHECK_CUDA(cudaMalloc(&d_output, output_size * sizeof(int)));

    // Launch with 3D configuration
    dim3 blockSize(4, 4, 2);  // 32 threads per block
    dim3 gridSize(2, 2, 2);   // 8 blocks

    printf("Grid configuration:\n");
    printf("  Grid: (%d, %d, %d) = %d blocks\n",
           gridSize.x, gridSize.y, gridSize.z,
           gridSize.x * gridSize.y * gridSize.z);
    printf("  Block: (%d, %d, %d) = %d threads\n",
           blockSize.x, blockSize.y, blockSize.z,
           blockSize.x * blockSize.y * blockSize.z);

    threadHierarchyKernel<<<gridSize, blockSize>>>(d_output);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, output_size * sizeof(int), cudaMemcpyDeviceToHost));

    // Display results for first few threads
    printf("Sample thread hierarchy information:\n");
    for (int i = 0; i < 5; i++) {
        int base = i * 10;
        if (h_output[base] != -1) {
            printf("  Thread %d: threadIdx=(%d,%d,%d), blockIdx=(%d,%d,%d), global_id=%d\n",
                   i, h_output[base], h_output[base+1], h_output[base+2],
                   h_output[base+3], h_output[base+4], h_output[base+5],
                   h_output[base+8]);
        }
    }

    CHECK_CUDA(cudaFree(d_output));
    printf("Thread hierarchy test completed!\n\n");
}

void testWarpOperations() {
    printf("=== Warp Operations Test ===\n");

    int device;
    CHECK_CUDA(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    if (prop.major < 3) {
        printf("Warp operations require compute capability 3.0+\n");
        printf("Current device: %d.%d\n", prop.major, prop.minor);
        printf("Skipping warp operations test\n\n");
        return;
    }

    const int n = 256;  // 8 warps
    std::vector<int> h_input(n);
    std::vector<int> h_output(n);

    // Initialize input
    for (int i = 0; i < n; i++) {
        h_input[i] = i % 100;
    }

    int *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output, n * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    warpLevelKernel<<<1, n>>>(d_input, d_output, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, n * sizeof(int), cudaMemcpyDeviceToHost));

    // Display warp results
    printf("Warp operation results (first 4 warps):\n");
    for (int warp = 0; warp < 4; warp++) {
        int base = warp * 4;
        printf("  Warp %d: sum=%d, all_pos=%d, any_neg=%d, count>50=%d\n",
               warp, h_output[base], h_output[base+1],
               h_output[base+2], h_output[base+3]);
    }

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    printf("Warp operations test completed!\n\n");
}

void testBlockCooperation() {
    printf("=== Block Cooperation Test ===\n");

    const int n = 10000;
    std::vector<float> h_input(n);
    std::vector<float> h_output(n);

    // Initialize input
    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>(i % 100 + 1);  // Avoid zeros
    }

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    int sharedMemSize = blockSize * sizeof(float);

    blockCooperativeKernel<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify normalization within first block
    float block_sum = 0.0f;
    for (int i = 0; i < blockSize && i < n; i++) {
        block_sum += h_output[i];
    }

    printf("Block cooperation results:\n");
    printf("  First block normalized sum: %.6f (should be ~1.0)\n", block_sum);
    printf("  Sample normalized values: %.6f, %.6f, %.6f\n",
           h_output[0], h_output[1], h_output[2]);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    printf("Block cooperation test completed!\n\n");
}

void testHardwareMapping() {
    printf("=== Hardware Mapping Test ===\n");

    const int num_blocks = 16;
    const int block_size = 256;

    std::vector<int> h_sm_info(num_blocks * 3);
    std::vector<int> h_warp_info(1000 * 4, -1);

    int *d_sm_info, *d_warp_info;
    CHECK_CUDA(cudaMalloc(&d_sm_info, num_blocks * 3 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_warp_info, 1000 * 4 * sizeof(int)));

    hardwareMappingKernel<<<num_blocks, block_size>>>(d_sm_info, d_warp_info);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_sm_info.data(), d_sm_info, num_blocks * 3 * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_warp_info.data(), d_warp_info, 1000 * 4 * sizeof(int), cudaMemcpyDeviceToHost));

    // Analyze SM distribution
    printf("SM distribution:\n");
    for (int block = 0; block < num_blocks; block++) {
        int base = block * 3;
        printf("  Block %2d: SM %d\n", block, h_sm_info[base]);
    }

    // Show warp mapping for first few warps
    printf("\nWarp mapping (first 8 warps):\n");
    for (int warp = 0; warp < 8 && h_warp_info[warp * 4] != -1; warp++) {
        int base = warp * 4;
        printf("  Warp %d: SM %d, Block %d, Local warp %d\n",
               warp, h_warp_info[base], h_warp_info[base+1], h_warp_info[base+2]);
    }

    CHECK_CUDA(cudaFree(d_sm_info));
    CHECK_CUDA(cudaFree(d_warp_info));
    printf("Hardware mapping test completed!\n\n");
}

int main() {
    printf("Advanced Thread Hierarchy Examples\n");
    printf("=================================\n\n");

    // Check device properties
    int device;
    CHECK_CUDA(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("\n");

    // Run all tests
    testThreadHierarchy();
    testWarpOperations();
    testBlockCooperation();
    testHardwareMapping();

    // Show optimization guidance
    LaunchOptimizer::demonstrateOccupancyOptimization();

    printf("All thread hierarchy examples completed successfully!\n");

    return 0;
}