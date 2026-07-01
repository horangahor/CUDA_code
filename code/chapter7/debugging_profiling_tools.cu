#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

// Error checking macro with debug info
#define CHECK_CUDA_DEBUG(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA DEBUG ERROR:\n"); \
        fprintf(stderr, "  File: %s\n", __FILE__); \
        fprintf(stderr, "  Line: %d\n", __LINE__); \
        fprintf(stderr, "  Function: %s\n", #call); \
        fprintf(stderr, "  Error: %s (%d)\n", cudaGetErrorString(error), error); \
        fprintf(stderr, "  Description: %s\n", cudaGetErrorName(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Debug logging macro
#define DEBUG_LOG(msg, ...) printf("[DEBUG] " msg "\n", ##__VA_ARGS__)

// =======================================================
// Debugging Tools Demonstration
// =======================================================

// Kernel with intentional bugs for debugging demonstration
__global__ void debugKernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Debug: Print thread info for first few threads
    if (idx < 5) {
        printf("GPU DEBUG: Thread %d processing element %d\n", idx, idx);
    }

    if (idx < n) {
        // Simulate processing
        output[idx] = sqrtf(input[idx] + 1.0f);

        // Debug: Check for invalid values
        if (isnan(output[idx]) || isinf(output[idx])) {
            printf("GPU WARNING: Invalid value at index %d: %f\n", idx, output[idx]);
        }
    }
}

// Kernel with bounds checking
__global__ void safeKernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Strict bounds checking
    if (idx >= n) {
        if (idx < n + 32) {  // Only warn for threads close to boundary
            printf("GPU BOUNDS: Thread %d exceeded bounds (n=%d)\n", idx, n);
        }
        return;
    }

    // Input validation
    if (input[idx] < 0.0f) {
        printf("GPU VALIDATION: Negative input at index %d: %f\n", idx, input[idx]);
        output[idx] = 0.0f;
        return;
    }

    output[idx] = sqrtf(input[idx] + 1.0f);
}

// =======================================================
// Performance Profiling Tools
// =======================================================

class CudaProfiler {
private:
    cudaEvent_t start_, stop_;
    bool timing_active_;

public:
    CudaProfiler() : timing_active_(false) {
        CHECK_CUDA_DEBUG(cudaEventCreate(&start_));
        CHECK_CUDA_DEBUG(cudaEventCreate(&stop_));
    }

    ~CudaProfiler() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void startTiming(const char* operation_name = "Operation") {
        DEBUG_LOG("Starting timing for: %s", operation_name);
        CHECK_CUDA_DEBUG(cudaEventRecord(start_));
        timing_active_ = true;
    }

    float stopTiming(const char* operation_name = "Operation") {
        if (!timing_active_) {
            DEBUG_LOG("Warning: stopTiming called without startTiming");
            return 0.0f;
        }

        CHECK_CUDA_DEBUG(cudaEventRecord(stop_));
        CHECK_CUDA_DEBUG(cudaEventSynchronize(stop_));

        float elapsed_ms;
        CHECK_CUDA_DEBUG(cudaEventElapsedTime(&elapsed_ms, start_, stop_));

        DEBUG_LOG("Completed timing for: %s (%.3f ms)", operation_name, elapsed_ms);
        timing_active_ = false;

        return elapsed_ms;
    }

    static void profileMemoryBandwidth(size_t bytes, float time_ms) {
        float bandwidth_gb_s = (bytes / 1e9) / (time_ms / 1000.0f);
        DEBUG_LOG("Memory bandwidth: %.2f GB/s", bandwidth_gb_s);
    }

    static void profileComputeThroughput(size_t operations, float time_ms) {
        float throughput_gops = (operations / 1e9) / (time_ms / 1000.0f);
        DEBUG_LOG("Compute throughput: %.2f GOPS", throughput_gops);
    }
};

// =======================================================
// Memory Debugging Tools
// =======================================================

class MemoryDebugger {
public:
    static void checkMemoryLeaks() {
        DEBUG_LOG("Checking for memory leaks...");

        size_t free_before, total_before;
        CHECK_CUDA_DEBUG(cudaMemGetInfo(&free_before, &total_before));

        // Allocate and free memory
        float* test_ptr;
        CHECK_CUDA_DEBUG(cudaMalloc(&test_ptr, 1024 * sizeof(float)));
        CHECK_CUDA_DEBUG(cudaFree(test_ptr));

        size_t free_after, total_after;
        CHECK_CUDA_DEBUG(cudaMemGetInfo(&free_after, &total_after));

        if (free_before != free_after) {
            DEBUG_LOG("WARNING: Potential memory leak detected!");
            DEBUG_LOG("  Free memory before: %zu bytes", free_before);
            DEBUG_LOG("  Free memory after: %zu bytes", free_after);
        } else {
            DEBUG_LOG("Memory leak check: PASSED");
        }
    }

    static void analyzeMemoryPattern(float* d_data, int n, const char* data_name) {
        DEBUG_LOG("Analyzing memory pattern for: %s", data_name);

        std::vector<float> h_sample(std::min(100, n));
        CHECK_CUDA_DEBUG(cudaMemcpy(h_sample.data(), d_data,
                                    h_sample.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost));

        // Check for patterns
        bool all_zero = true;
        bool has_nan = false;
        bool has_inf = false;
        float min_val = h_sample[0];
        float max_val = h_sample[0];

        for (float val : h_sample) {
            if (val != 0.0f) all_zero = false;
            if (isnan(val)) has_nan = true;
            if (isinf(val)) has_inf = true;
            min_val = fminf(min_val, val);
            max_val = fmaxf(max_val, val);
        }

        DEBUG_LOG("Memory pattern analysis:");
        DEBUG_LOG("  All zeros: %s", all_zero ? "Yes" : "No");
        DEBUG_LOG("  Contains NaN: %s", has_nan ? "Yes" : "No");
        DEBUG_LOG("  Contains Inf: %s", has_inf ? "Yes" : "No");
        DEBUG_LOG("  Range: [%.6f, %.6f]", min_val, max_val);
    }

    static void validatePointer(void* ptr, size_t size, const char* ptr_name) {
        DEBUG_LOG("Validating pointer: %s", ptr_name);

        if (ptr == nullptr) {
            DEBUG_LOG("ERROR: %s is null!", ptr_name);
            return;
        }

        // Check if pointer is device memory
        cudaPointerAttributes attributes;
        cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

        if (err == cudaSuccess) {
            DEBUG_LOG("Pointer validation for %s:", ptr_name);
            DEBUG_LOG("  Type: %s", (attributes.type == cudaMemoryTypeHost) ? "Host" :
                     (attributes.type == cudaMemoryTypeDevice) ? "Device" :
                     (attributes.type == cudaMemoryTypeManaged) ? "Managed" : "Unknown");
            DEBUG_LOG("  Device: %d", attributes.device);
        } else {
            DEBUG_LOG("WARNING: Could not get attributes for %s", ptr_name);
            cudaGetLastError(); // Clear error
        }
    }
};

// =======================================================
// Kernel Launch Debugger
// =======================================================

class KernelDebugger {
public:
    static void validateLaunchConfig(dim3 grid, dim3 block, const char* kernel_name) {
        DEBUG_LOG("Validating launch config for: %s", kernel_name);

        // Check device properties
        int device;
        CHECK_CUDA_DEBUG(cudaGetDevice(&device));

        cudaDeviceProp prop;
        CHECK_CUDA_DEBUG(cudaGetDeviceProperties(&prop, device));

        // Validate block size
        int total_threads = block.x * block.y * block.z;
        if (total_threads > prop.maxThreadsPerBlock) {
            DEBUG_LOG("ERROR: Block size (%d) exceeds maximum (%d)",
                     total_threads, prop.maxThreadsPerBlock);
        }

        // Validate grid size
        if (grid.x > prop.maxGridSize[0] ||
            grid.y > prop.maxGridSize[1] ||
            grid.z > prop.maxGridSize[2]) {
            DEBUG_LOG("ERROR: Grid size exceeds maximum");
            DEBUG_LOG("  Requested: (%d, %d, %d)", grid.x, grid.y, grid.z);
            DEBUG_LOG("  Maximum: (%d, %d, %d)",
                     prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        }

        DEBUG_LOG("Launch config validation: PASSED");
        DEBUG_LOG("  Grid: (%d, %d, %d)", grid.x, grid.y, grid.z);
        DEBUG_LOG("  Block: (%d, %d, %d) = %d threads", block.x, block.y, block.z, total_threads);
    }

    template<typename KernelFunc, typename... Args>
    static void debugLaunch(KernelFunc kernel, dim3 grid, dim3 block,
                           const char* kernel_name, Args... args) {
        DEBUG_LOG("Debug launching kernel: %s", kernel_name);

        validateLaunchConfig(grid, block, kernel_name);

        // Launch kernel
        kernel<<<grid, block>>>(args...);

        // Check for launch errors
        cudaError_t launch_error = cudaGetLastError();
        if (launch_error != cudaSuccess) {
            DEBUG_LOG("KERNEL LAUNCH ERROR: %s", cudaGetErrorString(launch_error));
            exit(EXIT_FAILURE);
        }

        // Synchronize and check for execution errors
        cudaError_t sync_error = cudaDeviceSynchronize();
        if (sync_error != cudaSuccess) {
            DEBUG_LOG("KERNEL EXECUTION ERROR: %s", cudaGetErrorString(sync_error));
            exit(EXIT_FAILURE);
        }

        DEBUG_LOG("Kernel %s completed successfully", kernel_name);
    }
};

// =======================================================
// Performance Benchmarking Suite
// =======================================================

class PerformanceBenchmark {
public:
    static void benchmarkKernel(const char* kernel_name, int iterations = 100) {
        DEBUG_LOG("Benchmarking kernel: %s (%d iterations)", kernel_name, iterations);

        const int n = 1000000;
        std::vector<float> h_input(n, 1.0f);
        std::vector<float> h_output(n);

        float *d_input, *d_output;
        CHECK_CUDA_DEBUG(cudaMalloc(&d_input, n * sizeof(float)));
        CHECK_CUDA_DEBUG(cudaMalloc(&d_output, n * sizeof(float)));

        CHECK_CUDA_DEBUG(cudaMemcpy(d_input, h_input.data(),
                                    n * sizeof(float), cudaMemcpyHostToDevice));

        CudaProfiler profiler;

        // Warmup
        debugKernel<<<(n + 255) / 256, 256>>>(d_input, d_output, n);
        CHECK_CUDA_DEBUG(cudaDeviceSynchronize());

        // Benchmark
        profiler.startTiming(kernel_name);
        for (int i = 0; i < iterations; i++) {
            debugKernel<<<(n + 255) / 256, 256>>>(d_input, d_output, n);
        }
        CHECK_CUDA_DEBUG(cudaDeviceSynchronize());
        float total_time = profiler.stopTiming(kernel_name);

        float avg_time = total_time / iterations;
        size_t operations = static_cast<size_t>(n) * iterations;

        DEBUG_LOG("Benchmark results for %s:", kernel_name);
        DEBUG_LOG("  Total time: %.3f ms", total_time);
        DEBUG_LOG("  Average time: %.3f ms", avg_time);
        DEBUG_LOG("  Operations: %zu", operations);

        CudaProfiler::profileComputeThroughput(operations, total_time);

        CHECK_CUDA_DEBUG(cudaFree(d_input));
        CHECK_CUDA_DEBUG(cudaFree(d_output));
    }

    static void memoryBandwidthTest() {
        DEBUG_LOG("Starting memory bandwidth test...");

        const size_t sizes[] = {
            1 * 1024 * 1024,    // 1MB
            10 * 1024 * 1024,   // 10MB
            100 * 1024 * 1024   // 100MB
        };

        for (size_t size_bytes : sizes) {
            size_t n_elements = size_bytes / sizeof(float);

            std::vector<float> h_data(n_elements, 1.0f);
            float* d_data;
            CHECK_CUDA_DEBUG(cudaMalloc(&d_data, size_bytes));

            CudaProfiler profiler;

            // Test H2D transfer
            profiler.startTiming("H2D Transfer");
            CHECK_CUDA_DEBUG(cudaMemcpy(d_data, h_data.data(), size_bytes,
                                        cudaMemcpyHostToDevice));
            float h2d_time = profiler.stopTiming("H2D Transfer");

            // Test D2H transfer
            profiler.startTiming("D2H Transfer");
            CHECK_CUDA_DEBUG(cudaMemcpy(h_data.data(), d_data, size_bytes,
                                        cudaMemcpyDeviceToHost));
            float d2h_time = profiler.stopTiming("D2H Transfer");

            DEBUG_LOG("Memory bandwidth test (%.1f MB):", size_bytes / 1e6);
            CudaProfiler::profileMemoryBandwidth(size_bytes, h2d_time);
            CudaProfiler::profileMemoryBandwidth(size_bytes, d2h_time);

            CHECK_CUDA_DEBUG(cudaFree(d_data));
        }
    }
};

// =======================================================
// Test Functions
// =======================================================

void testDebuggingTools() {
    DEBUG_LOG("=== Testing Debugging Tools ===");

    const int n = 10000;
    std::vector<float> h_input(n);
    std::vector<float> h_output(n);

    // Initialize with some negative values for testing
    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>(i - 500);  // Some negative values
    }

    float *d_input, *d_output;
    CHECK_CUDA_DEBUG(cudaMalloc(&d_input, n * sizeof(float)));
    CHECK_CUDA_DEBUG(cudaMalloc(&d_output, n * sizeof(float)));

    MemoryDebugger::validatePointer(d_input, n * sizeof(float), "d_input");
    MemoryDebugger::validatePointer(d_output, n * sizeof(float), "d_output");

    CHECK_CUDA_DEBUG(cudaMemcpy(d_input, h_input.data(),
                                n * sizeof(float), cudaMemcpyHostToDevice));

    MemoryDebugger::analyzeMemoryPattern(d_input, n, "input_data");

    // Test kernel with bounds checking
    KernelDebugger::debugLaunch(safeKernel, dim3((n + 255) / 256), dim3(256),
                                "safeKernel", d_input, d_output, n);

    MemoryDebugger::analyzeMemoryPattern(d_output, n, "output_data");

    CHECK_CUDA_DEBUG(cudaFree(d_input));
    CHECK_CUDA_DEBUG(cudaFree(d_output));

    MemoryDebugger::checkMemoryLeaks();

    DEBUG_LOG("Debugging tools test completed!");
}

void testProfilingTools() {
    DEBUG_LOG("=== Testing Profiling Tools ===");

    PerformanceBenchmark::benchmarkKernel("debugKernel", 50);
    PerformanceBenchmark::memoryBandwidthTest();

    DEBUG_LOG("Profiling tools test completed!");
}

void demonstrateNsightIntegration() {
    DEBUG_LOG("=== Nsight Integration Demonstration ===");

    DEBUG_LOG("For advanced profiling, use these Nsight tools:");
    DEBUG_LOG("1. nsys profile ./your_program");
    DEBUG_LOG("   - System-wide performance analysis");
    DEBUG_LOG("   - Timeline view of CPU and GPU activities");
    DEBUG_LOG("   - Memory transfer analysis");

    DEBUG_LOG("2. ncu ./your_program");
    DEBUG_LOG("   - Detailed kernel analysis");
    DEBUG_LOG("   - Warp execution efficiency");
    DEBUG_LOG("   - Memory throughput metrics");

    DEBUG_LOG("3. cuda-memcheck ./your_program");
    DEBUG_LOG("   - Memory error detection");
    DEBUG_LOG("   - Out-of-bounds access detection");
    DEBUG_LOG("   - Race condition detection");

    DEBUG_LOG("Example command line usage:");
    DEBUG_LOG("  nsys profile --trace=cuda,nvtx ./program");
    DEBUG_LOG("  ncu --metrics=sm__throughput.avg.pct_of_peak_sustained_elapsed ./program");
    DEBUG_LOG("  cuda-memcheck --tool=racecheck ./program");
}

void showDebugBestPractices() {
    DEBUG_LOG("=== Debug Best Practices ===");

    DEBUG_LOG("1. Error Checking:");
    DEBUG_LOG("   - Check every CUDA API call");
    DEBUG_LOG("   - Use meaningful error messages");
    DEBUG_LOG("   - Check both launch and execution errors");

    DEBUG_LOG("2. Kernel Debugging:");
    DEBUG_LOG("   - Use printf for simple debugging");
    DEBUG_LOG("   - Add bounds checking");
    DEBUG_LOG("   - Validate input parameters");

    DEBUG_LOG("3. Memory Debugging:");
    DEBUG_LOG("   - Initialize all memory");
    DEBUG_LOG("   - Check for leaks");
    DEBUG_LOG("   - Validate pointer types");

    DEBUG_LOG("4. Performance Profiling:");
    DEBUG_LOG("   - Use CUDA events for accurate timing");
    DEBUG_LOG("   - Measure both compute and memory performance");
    DEBUG_LOG("   - Profile on target hardware");

    DEBUG_LOG("5. Tools Integration:");
    DEBUG_LOG("   - Use Nsight for visual profiling");
    DEBUG_LOG("   - Automate testing with scripts");
    DEBUG_LOG("   - Set up continuous integration");
}

int main() {
    printf("CUDA Debugging and Profiling Tools Examples\n");
    printf("===========================================\n\n");

    // Device information
    int device;
    CHECK_CUDA_DEBUG(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CHECK_CUDA_DEBUG(cudaGetDeviceProperties(&prop, device));

    DEBUG_LOG("Device: %s (Compute %d.%d)", prop.name, prop.major, prop.minor);
    DEBUG_LOG("Memory: %.2f GB", prop.totalGlobalMem / 1e9);

    // Run all tests
    testDebuggingTools();
    testProfilingTools();
    demonstrateNsightIntegration();
    showDebugBestPractices();

    DEBUG_LOG("All debugging and profiling examples completed successfully!");

    return 0;
}