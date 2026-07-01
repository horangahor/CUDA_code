// 06-branch-divergence.cu
// 슬라이드: part1/chapter5/06-branch-divergence.md
// Divergent vs warp-aligned vs branchless, divergence 측정
//
// 빌드: nvcc -O3 06-branch-divergence.cu -o branch_divergence
// 실행: ./branch_divergence

#include <cstdio>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────
// ❌ Even/Odd split — warp 내부에서 50% 분기
// ─────────────────────────────────────────────────────────────
__global__ void divergentKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (idx % 2 == 0) {
        data[idx] = sinf(data[idx]) + cosf(data[idx]);
    } else {
        data[idx] = data[idx] * 2.0f;
    }
}

// ─────────────────────────────────────────────────────────────
// ✅ Warp-aligned — 같은 워프는 모두 같은 경로
// ─────────────────────────────────────────────────────────────
__global__ void convergentKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int warpGroup = (threadIdx.x / 32) % 2;
    if (warpGroup == 0) {
        data[idx] = sinf(data[idx]) + cosf(data[idx]);
    } else {
        data[idx] = data[idx] * 2.0f;
    }
}

// ─────────────────────────────────────────────────────────────
// ✅ Branchless — 산술 연산으로 분기 제거
// ─────────────────────────────────────────────────────────────
__device__ unsigned char enhance_dark(unsigned char p) {
    int v = (int)p * 2; return (unsigned char)(v > 255 ? 255 : v);
}

__global__ void branchlessFilter(const unsigned char* in,
                                 unsigned char* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned char p = in[idx];
    int is_dark = (p < 128);                          // 0 or 1
    unsigned char enhanced = enhance_dark(p);
    out[idx] = (unsigned char)(is_dark * enhanced + (1 - is_dark) * p);
}

// ─────────────────────────────────────────────────────────────
// Divergence 런타임 측정 — __activemask / __popc
// ─────────────────────────────────────────────────────────────
__global__ void detectDivergence(const float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (data[idx] > 0.0f) {
        unsigned mask = __activemask();
        int active   = __popc(mask);
        float eff    = (float)active / 32.0f;
        if ((threadIdx.x % 32) == 0 && eff < 0.5f) {
            printf("warp %d: divergence efficiency %.0f%%\n",
                   threadIdx.x / 32, eff * 100.0f);
        }
    }
}

int main() {
    constexpr int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float* d_data;
    cudaMalloc(&d_data, bytes);

    unsigned char* d_img;
    unsigned char* d_out;
    cudaMalloc(&d_img, N);
    cudaMalloc(&d_out, N);

    int block = 256;
    int grid  = (N + block - 1) / block;

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    cudaEventRecord(s);
    divergentKernel<<<grid, block>>>(d_data, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float t_div; cudaEventElapsedTime(&t_div, s, e);

    cudaEventRecord(s);
    convergentKernel<<<grid, block>>>(d_data, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float t_con; cudaEventElapsedTime(&t_con, s, e);

    printf("divergent  : %.3f ms\n", t_div);
    printf("convergent : %.3f ms (speedup x%.2f)\n", t_con, t_div / t_con);

    branchlessFilter<<<grid, block>>>(d_img, d_out, N);
    detectDivergence<<<grid, block>>>(d_data, N);
    cudaDeviceSynchronize();

    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(d_data); cudaFree(d_img); cudaFree(d_out);
    return 0;
}
