// 02-thread-indexing.cu
// 슬라이드: part1/chapter4/02-indexing-and-configuration.md
// 1D / 2D / 3D 인덱싱, dim3, grid 크기 올림, 경계 체크, grid-stride
//
// 빌드: nvcc 02-thread-indexing.cu -o thread_indexing
// 실행: ./thread_indexing

#include <cstdio>
#include <cuda_runtime.h>

#define DIVUP(n, d) (((n) + (d) - 1) / (d))

// ─────────────────────────────────────────────────────────────
// 1D — 한 줄짜리 패턴
// ─────────────────────────────────────────────────────────────
__global__ void kernel_1d(float* A, float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) C[tid] = A[tid] + B[tid];
}

// ─────────────────────────────────────────────────────────────
// 2D — 이미지/행렬, row-major 선형화
// ─────────────────────────────────────────────────────────────
__global__ void kernel_2d(float* data, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;            // 경계 체크
    int idx = y * width + x;                          // 2D → 1D
    data[idx] *= 2.0f;
}

// ─────────────────────────────────────────────────────────────
// 3D — 볼륨 데이터
// ─────────────────────────────────────────────────────────────
__global__ void kernel_3d(float* volume, int W, int H, int D) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;
    int idx = z * (W * H) + y * W + x;
    volume[idx] = sqrtf(volume[idx]);
}

// ─────────────────────────────────────────────────────────────
// Grid-Stride Loop — 임의 크기 데이터 처리
// ─────────────────────────────────────────────────────────────
__global__ void kernel_grid_stride(float* A, float* B, float* C, int N) {
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    constexpr int N = 1 << 16;
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // 1D 호출
    int blockSize = 256;
    int gridSize  = DIVUP(N, blockSize);
    kernel_1d<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // 2D 호출
    int W = 512, H = 512;
    float* d_img;
    cudaMalloc(&d_img, W * H * sizeof(float));
    dim3 block2(16, 16);                              // 16×16 = 256 threads
    dim3 grid2(DIVUP(W, 16), DIVUP(H, 16));
    kernel_2d<<<grid2, block2>>>(d_img, W, H);

    // 3D 호출
    int D = 32;
    float* d_vol;
    cudaMalloc(&d_vol, W * H * D * sizeof(float));
    dim3 block3(8, 8, 8);                             // 512 threads
    dim3 grid3(DIVUP(W, 8), DIVUP(H, 8), DIVUP(D, 8));
    kernel_3d<<<grid3, block3>>>(d_vol, W, H, D);

    // Grid-Stride 호출 — SM 수 기반
    int numSMs = 0;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    kernel_grid_stride<<<32 * numSMs, blockSize>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_img); cudaFree(d_vol);
    return 0;
}
