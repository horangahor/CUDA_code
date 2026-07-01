// 09-warp-primitives.cu
// 슬라이드: part1/chapter5/09-warp-level-primitives.md
// Vote: __all_sync / __any_sync / __ballot_sync 활용 패턴
//
// 빌드: nvcc -arch=sm_87 -O3 09-warp-primitives.cu -o warp_primitives  # Jetson Orin Nano
// 실행: ./warp_primitives

#include <cstdio>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────
// Vote 함수 기본 — 같은 조건에 대해 3가지 결과 비교
// ─────────────────────────────────────────────────────────────
__global__ void vote_basics() {
    int tid       = threadIdx.x;
    unsigned mask = __activemask();
    bool cond     = (tid % 4 == 0);

    bool allTrue      = __all_sync(mask, cond);
    bool anyTrue      = __any_sync(mask, cond);
    unsigned ballot   = __ballot_sync(mask, cond);

    if (tid == 0) {
        printf("all=%d any=%d ballot=0x%08x (popc=%d)\n",
               allTrue, anyTrue, ballot, __popc(ballot));
    }
}

// ─────────────────────────────────────────────────────────────
// Early termination — __all_sync로 워프 전체 종료
// ─────────────────────────────────────────────────────────────
__device__ float computeError(float v) { return v * 0.5f; }
__device__ void  updateState(float* v) { *v *= 0.99f; }

__global__ void converge_loop(float* state, int maxIters, float thr) {
    int tid       = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned mask = __activemask();
    float v       = state[tid];

    for (int it = 0; it < maxIters; ++it) {
        float err = computeError(v);
        if (__all_sync(mask, err < thr)) break;     // 전부 수렴 → 워프 종료
        updateState(&v);
    }
    state[tid] = v;
}

// ─────────────────────────────────────────────────────────────
// 활용 1: 전부 유효한가 (fast path 선택)
// ─────────────────────────────────────────────────────────────
__device__ void fastPath()  { /* ... */ }
__device__ void slowPath()  { /* ... */ }

__global__ void validity_check(const float* d, int n) {
    int idx       = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned mask = __activemask();
    bool valid    = (d[idx] >= 0.0f);
    if (__all_sync(mask, valid)) fastPath();
    else                         slowPath();
}

// ─────────────────────────────────────────────────────────────
// 활용 2: 작업 분배 — ballot으로 helper 식별
// ─────────────────────────────────────────────────────────────
__global__ void work_distribution(const int* workload, int capacity, int n) {
    int idx       = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned mask = __activemask();
    bool overload = (workload[idx] > capacity);
    unsigned helpNeeded = __ballot_sync(mask, overload);

    if ((threadIdx.x % 32) == 0 && helpNeeded != 0) {
        printf("warp %d: %d threads need help (mask=0x%08x)\n",
               (blockIdx.x * blockDim.x + threadIdx.x) / 32,
               __popc(helpNeeded), helpNeeded);
    }
}

int main() {
    vote_basics<<<1, 32>>>();

    constexpr int N = 1024;
    float* d_state;
    cudaMalloc(&d_state, N * sizeof(float));
    converge_loop<<<(N + 31) / 32, 32>>>(d_state, 100, 0.01f);

    float* d_d;
    cudaMalloc(&d_d, N * sizeof(float));
    validity_check<<<(N + 31) / 32, 32>>>(d_d, N);

    int* d_w;
    cudaMalloc(&d_w, N * sizeof(int));
    work_distribution<<<(N + 31) / 32, 32>>>(d_w, 100, N);

    cudaDeviceSynchronize();
    cudaFree(d_state); cudaFree(d_d); cudaFree(d_w);
    return 0;
}
