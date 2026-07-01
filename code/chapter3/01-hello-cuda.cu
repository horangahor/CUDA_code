// 01-hello-cuda.cu
// 슬라이드: part1/chapter3/01-hello-cuda.md
// 첫 CUDA 프로그램 + cudaEvent 시간 측정
//
// 빌드: nvcc 01-hello-cuda.cu -o hello_cuda
//      nvcc -arch=sm_87 -O3 01-hello-cuda.cu -o hello_cuda    # Jetson Orin Nano
// 실행: ./hello_cuda

#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_kernel(int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        printf("Hello from GPU thread %d in block %d!\n",
               threadIdx.x, blockIdx.x);
    }
}

int main() {
    // 1. 기본 실행
    hello_kernel<<<2, 4>>>(8);   // 2블록 × 4스레드 = 8개 출력
    cudaDeviceSynchronize();     // GPU 완료 대기

    // 2. cudaEvent로 GPU 실행 시간 측정
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    hello_kernel<<<2, 4>>>(8);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel time: %.3f ms\n", ms);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
