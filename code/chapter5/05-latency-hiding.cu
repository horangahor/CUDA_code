// 05-latency-hiding.cu
// 슬라이드: part1/chapter5/05-latency-hiding.md
// Double Buffering, ILP, Coalescing 패턴
//
// 빌드: nvcc -O3 05-latency-hiding.cu -o latency_hiding
// 실행: ./latency_hiding

#include <cstdio>
#include <cuda_runtime.h>

__device__ float expensive_compute(float x) {
    float v = x;
    #pragma unroll 16
    for (int i = 0; i < 16; ++i) v = v * 1.001f + 0.001f;
    return v;
}
__device__ float more_compute(float x) { return x * 0.5f; }
__device__ void  process(float* sm)    { sm[threadIdx.x] *= 1.5f; }

// ─────────────────────────────────────────────────────────────
// Double Buffering — 다음 블록 로드와 현재 블록 처리 중첩
// ─────────────────────────────────────────────────────────────
__global__ void doubleBuffering(const float* g, float* out, int iters) {
    __shared__ float buffer[2][256];
    int tid = threadIdx.x;
    int current = 0;

    // 첫 번째 블록 로드
    buffer[0][tid] = g[tid];
    __syncthreads();

    for (int iter = 1; iter < iters; ++iter) {
        int next = 1 - current;
        // 다음 블록 비동기 로드 + 현재 블록 처리
        buffer[next][tid] = g[iter * 256 + tid];
        process(buffer[current]);
        __syncthreads();
        current = next;
    }
    out[tid] = buffer[current][tid];
}

// ─────────────────────────────────────────────────────────────
// ILP 비교 — 직렬 의존 vs 독립 로드
// ─────────────────────────────────────────────────────────────
__global__ void low_ilp(const float* data, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float a = data[idx];        // LD
    float b = a * 2.0f;         // a 의존
    float c = b + 1.0f;         // b 의존
    out[idx] = c;
}

__global__ void high_ilp(const float* data, const float* other,
                         const float* third, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float a = data[idx];        // 3개 독립 로드 → 동시 in-flight
    float b = other[idx];
    float c = third[idx];
    out[idx] = a + b + c;
}

// ─────────────────────────────────────────────────────────────
// 메모리 대기 중 독립 연산 중첩
// ─────────────────────────────────────────────────────────────
__global__ void overlapKernel(const float* g, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float remote = g[idx];                // 1. 메모리 요청
    float local  = expensive_compute((float)idx);  // 2. 대기 중 연산
    float other  = more_compute((float)idx);       // 3. 추가 독립 연산
    out[idx] = remote + local + other;             // 4. 도착 후 사용
}

int main() {
    constexpr int N = 1 << 20;
    float *d_g, *d_out, *d_other, *d_third;
    cudaMalloc(&d_g,     N * sizeof(float));
    cudaMalloc(&d_out,   N * sizeof(float));
    cudaMalloc(&d_other, N * sizeof(float));
    cudaMalloc(&d_third, N * sizeof(float));

    int block = 256;
    int grid  = (N + block - 1) / block;

    doubleBuffering<<<1, 256>>>(d_g, d_out, 16);
    low_ilp<<<grid, block>>>(d_g, d_out, N);
    high_ilp<<<grid, block>>>(d_g, d_other, d_third, d_out, N);
    overlapKernel<<<grid, block>>>(d_g, d_out, N);
    cudaDeviceSynchronize();

    cudaFree(d_g); cudaFree(d_out); cudaFree(d_other); cudaFree(d_third);
    return 0;
}
