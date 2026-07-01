// 03-race-condition.cu
// 슬라이드: part0/chapter1/04-sync-async-blocking.md
// 동기화 없는 공유 변수 접근 → race condition. atomicAdd로 해결.
//
// 빌드: nvcc 03-race-condition.cu -o race_condition
// 실행: ./race_condition

#include <cstdio>
#include <cuda_runtime.h>

// ❌ Race condition: read-modify-write 경쟁
__global__ void racy_increment(int* counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        (*counter)++;             // 비원자적 — 손실 발생
    }
}

// ✅ atomicAdd로 원자적 증가
__global__ void atomic_increment(int* counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(counter, 1);
    }
}

int main() {
    constexpr int N = 1 << 16;
    int* d_counter;
    cudaMalloc(&d_counter, sizeof(int));

    int zero = 0, h_counter = 0;

    cudaMemcpy(d_counter, &zero, sizeof(int), cudaMemcpyHostToDevice);
    racy_increment<<<(N + 255) / 256, 256>>>(d_counter, N);
    cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Racy   counter = %d (expected %d)\n", h_counter, N);

    cudaMemcpy(d_counter, &zero, sizeof(int), cudaMemcpyHostToDevice);
    atomic_increment<<<(N + 255) / 256, 256>>>(d_counter, N);
    cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Atomic counter = %d (expected %d)\n", h_counter, N);

    cudaFree(d_counter);
    return 0;
}
