// 04-vector-addition.cu
// 슬라이드: part1/chapter3/04-vector-addition.md
// 6단계 패턴 + Grid-Stride Loop
//
// 빌드: nvcc -O3 04-vector-addition.cu -o vector_addition
// 실행: ./vector_addition

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────
// 기본 커널: 한 스레드가 한 원소 처리
// ─────────────────────────────────────────────────────────────
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

// ─────────────────────────────────────────────────────────────
// Grid-Stride Loop: 한 스레드가 여러 원소 처리
// ─────────────────────────────────────────────────────────────
__global__ void vectorAddGridStride(const float* A, const float* B, float* C, int N) {
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

void cpu_vectorAdd(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) C[i] = A[i] + B[i];
}

int main() {
    constexpr int N = 1 << 20;          // 1M 요소
    size_t size = N * sizeof(float);

    // Step 1: Host 메모리 할당 + 초기화
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)rand() / (float)RAND_MAX;
        h_B[i] = (float)rand() / (float)RAND_MAX;
    }

    // Step 2: Device 메모리 할당
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Step 3: Host → Device 복사
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Step 4: 커널 실행 — 기본 패턴
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Step 4-b: Grid-Stride Loop — SM 수에 맞춘 작은 grid
    int numSMs = 0;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    vectorAddGridStride<<<32 * numSMs, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Step 5: Device → Host 복사
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 검증
    float* h_ref = (float*)malloc(size);
    cpu_vectorAdd(h_A, h_B, h_ref, N);
    int errs = 0;
    for (int i = 0; i < N; ++i)
        if (h_C[i] != h_ref[i]) errs++;
    printf("N=%d  errors=%d\n", N, errs);

    // Step 6: 메모리 해제
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return 0;
}
