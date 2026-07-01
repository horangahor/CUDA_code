// 01-warp-basics.cu
// 슬라이드: part1/chapter5/01-warp-as-simt-unit.md
// Warp / Lane ID, 블록 크기에 따른 효율 비교
//
// 빌드: nvcc -O3 01-warp-basics.cu -o warp_basics
// 실행: ./warp_basics

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

// Warp ID와 Lane ID 출력
__global__ void understandWarp() {
    int tid    = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;
    if (tid < 64) {
        printf("Thread %3d: Warp %d, Lane %2d\n", tid, warpId, laneId);
    }
}

// 단순 메모리 바운드 커널 — 블록 크기 영향 측정용
__global__ void efficientKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = sqrtf(data[idx]);
}

int main() {
    // Warp / Lane 시연
    understandWarp<<<1, 64>>>();
    cudaDeviceSynchronize();

    // 블록 크기별 효율 비교
    constexpr int N = 10'000'000;
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    int sizes[] = {32, 64, 128, 256, 100, 200};        // 32의 배수 vs 비배수
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    printf("\n%4s | %-7s | %-9s\n", "Blk", "Eff", "Time(ms)");
    for (int i = 0; i < 6; ++i) {
        int block = sizes[i];
        int grid  = (N + block - 1) / block;

        cudaEventRecord(s);
        efficientKernel<<<grid, block>>>(d_data, N);
        cudaEventRecord(e); cudaEventSynchronize(e);

        float ms; cudaEventElapsedTime(&ms, s, e);
        // 32의 배수가 아니면 마지막 워프가 부분 활용
        float eff = (block % 32 == 0)
            ? 100.0f
            : (float)block / ((block / 32 + 1) * 32) * 100.0f;
        printf("%4d | %5.1f%% | %.3f\n", block, eff, ms);
    }

    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(d_data);

    // (참고) cudaOccupancyMaxPotentialBlockSize로 자동 추천
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, efficientKernel, 0, 0);
    printf("\nSuggested blockSize = %d, minGridSize = %d\n",
           blockSize, minGridSize);
    return 0;
}
