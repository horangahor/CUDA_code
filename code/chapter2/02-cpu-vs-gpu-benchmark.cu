// 02-cpu-vs-gpu-benchmark.cu
// 슬라이드: part0/chapter2/01-cpu-vs-gpu.md, 08-performance-characteristics.md
// 메모리 바운드 커널의 CPU vs GPU 처리량 비교
//
// 빌드: nvcc -O3 02-cpu-vs-gpu-benchmark.cu -o cpu_vs_gpu
// 실행: ./cpu_vs_gpu

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

void cpu_saxpy(float* y, const float* x, float a, int n) {
    for (int i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}

__global__ void gpu_saxpy(float* y, const float* x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    constexpr int N = 1 << 24;        // 16M elements
    size_t bytes = N * sizeof(float);

    float* h_x = (float*)malloc(bytes);
    float* h_y = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) { h_x[i] = 1.0f; h_y[i] = 2.0f; }

    // CPU
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_saxpy(h_y, h_x, 3.0f, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // GPU
    float *d_x, *d_y;
    cudaMalloc(&d_x, bytes); cudaMalloc(&d_y, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    int block = 256;
    int grid  = (N + block - 1) / block;

    cudaEventRecord(start);
    gpu_saxpy<<<grid, block>>>(d_y, d_x, 3.0f, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    double bytes_moved = 3.0 * bytes;     // x read, y read+write
    double cpu_bw = bytes_moved / (cpu_ms * 1e-3) / 1e9;
    double gpu_bw = bytes_moved / (gpu_ms * 1e-3) / 1e9;

    printf("N = %d (%.0f MB working set)\n", N, bytes / 1.0e6);
    printf("CPU : %7.2f ms  | %5.1f GB/s\n", cpu_ms, cpu_bw);
    printf("GPU : %7.2f ms  | %5.1f GB/s  | speedup x%.1f\n",
           gpu_ms, gpu_bw, cpu_ms / gpu_ms);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_x); cudaFree(d_y);
    free(h_x); free(h_y);
    return 0;
}
