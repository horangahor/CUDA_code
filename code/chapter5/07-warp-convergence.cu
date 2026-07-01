// 07-warp-convergence.cu
// 슬라이드: part1/chapter5/07-warp-convergence.md
// 중첩 분기, Early exit (vote 기반), 단계별 수렴, 디버깅
//
// 빌드: nvcc -arch=sm_87 -O3 07-warp-convergence.cu -o warp_convergence  # Jetson Orin Nano
// 실행: ./warp_convergence

#include <cstdio>
#include <cuda_runtime.h>

__device__ void workA1()       { /* ... */ }
__device__ void workA2()       { /* ... */ }
__device__ void workA_common() { /* ... */ }
__device__ void workB()        { /* ... */ }
__device__ void finalWork()    { /* ... */ }

// ─────────────────────────────────────────────────────────────
// 중첩 분기와 명시적 수렴
// ─────────────────────────────────────────────────────────────
__global__ void nestedBranches() {
    int tid = threadIdx.x;
    unsigned mask = __activemask();

    if (tid < 16) {                     // Level 1
        if (tid < 8) {                  // Level 2 (A1)
            workA1();
        } else {                        // (A2)
            workA2();
        }
        __syncwarp(__activemask());     // A1, A2 수렴
        workA_common();
    } else {
        workB();
    }
    __syncwarp(mask);                   // 전체 수렴
    finalWork();
}

// ─────────────────────────────────────────────────────────────
// Early Exit — 일부만 종료 vs warp vote로 전체 종료
// ─────────────────────────────────────────────────────────────
__device__ bool checkExit(int tid) { return tid > 28; }

__global__ void earlyExit_partial() {
    int tid = threadIdx.x;
    if (checkExit(tid)) return;         // 일부만 종료 — warp는 계속 점유
    // 남은 스레드만 작업, 종료된 lane 슬롯은 낭비
}

__global__ void earlyExit_warpVote() {
    int tid = threadIdx.x;
    bool shouldExit = checkExit(tid);
    if (__all_sync(__activemask(), shouldExit)) {
        return;                         // 전체 워프가 동의해야만 종료
    }
}

// ─────────────────────────────────────────────────────────────
// 가변 루프 수렴 — __reduce_max_sync (Ampere+)
// ─────────────────────────────────────────────────────────────
__device__ void doWork(int i) { (void)i; }

__global__ void variableLoop_balanced(const int* counts) {
    int tid          = threadIdx.x;
    int workCount    = counts[tid];
    unsigned mask    = __activemask();
#if __CUDA_ARCH__ >= 800
    int maxWork = __reduce_max_sync(mask, workCount);
#else
    int maxWork = workCount;
    for (int off = 16; off > 0; off /= 2) {
        int v = __shfl_down_sync(mask, maxWork, off);
        maxWork = v > maxWork ? v : maxWork;
    }
#endif
    for (int i = 0; i < maxWork; ++i) {
        if (i < workCount) doWork(i);
    }
}

// ─────────────────────────────────────────────────────────────
// 수렴 디버깅 — 분기 전후 마스크 확인
// ─────────────────────────────────────────────────────────────
__device__ void debugConvergence(const char* label) {
    unsigned mask = __activemask();
    int active = __popc(mask);
    if ((threadIdx.x % 32) == 0) {
        printf("%-12s mask=0x%08x active=%d/32\n", label, mask, active);
    }
}

__global__ void debugExample() {
    debugConvergence("initial");
    if (threadIdx.x < 16) {
        debugConvergence("if");
    } else {
        debugConvergence("else");
    }
    __syncwarp();
    debugConvergence("after");
}

int main() {
    nestedBranches<<<1, 32>>>();
    earlyExit_partial<<<1, 32>>>();
    earlyExit_warpVote<<<1, 32>>>();

    int* d_counts;
    cudaMalloc(&d_counts, 32 * sizeof(int));
    int h[32]; for (int i = 0; i < 32; ++i) h[i] = i * 3;
    cudaMemcpy(d_counts, h, sizeof(h), cudaMemcpyHostToDevice);
    variableLoop_balanced<<<1, 32>>>(d_counts);

    debugExample<<<1, 32>>>();
    cudaDeviceSynchronize();
    cudaFree(d_counts);
    return 0;
}
