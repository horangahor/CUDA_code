// 03-matrix-multiply.cu
// 슬라이드: part1/chapter4/03-matrix-multiply.md
// 행렬 곱셈 3가지 방법: 2D 인덱싱 / 1D 선형화 / Grid-Stride Loop
//
// 빌드: nvcc -O3 03-matrix-multiply.cu -o matrix_multiply
// 실행: ./matrix_multiply

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────
// 방법 1: 2D Indexing — Thread(row, col)이 C[row][col] 계산
// ─────────────────────────────────────────────────────────────
__global__ void matMul2D(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// ─────────────────────────────────────────────────────────────
// 방법 2: 1D 선형화 — N×N을 길이 N²의 1D로 처리
// ─────────────────────────────────────────────────────────────
__global__ void matMul1D(const float* A, const float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N * N) {
        int row = tid / N;
        int col = tid % N;
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[tid] = sum;
    }
}

// ─────────────────────────────────────────────────────────────
// 방법 3: Grid-Stride Loop — 고정 grid로 임의 크기 처리
// ─────────────────────────────────────────────────────────────
__global__ void matMulStride(const float* A, const float* B, float* C, int N) {
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < N * N; i += stride) {
        int row = i / N;
        int col = i % N;
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[i] = sum;
    }
}

int main() {
    constexpr int N = 1024;
    size_t bytes = N * N * sizeof(float);

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    // 방법 1: 2D Indexing
    {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (N + 15) / 16);
        cudaEventRecord(s);
        matMul2D<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        printf("matMul2D     : %7.2f ms\n", ms);
    }

    // 방법 2: 1D 선형화
    {
        int block = 256;
        int grid  = (N * N + block - 1) / block;
        cudaEventRecord(s);
        matMul1D<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        printf("matMul1D     : %7.2f ms\n", ms);
    }

    // 방법 3: Grid-Stride Loop
    {
        int block = 256;
        int numSMs = 0;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
        int grid = 32 * numSMs;
        cudaEventRecord(s);
        matMulStride<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        printf("matMulStride : %7.2f ms\n", ms);
    }

    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B);
    return 0;
}
