// 08-syncwarp-activemask.cu
// 슬라이드: part1/chapter5/08-syncwarp-activemask.md
// __activemask 비트 패턴, race 방지, 동적 마스크 관리, rank 계산
//
// 빌드: nvcc -arch=sm_87 -O3 08-syncwarp-activemask.cu -o syncwarp_activemask  # Jetson Orin Nano
// 실행: ./syncwarp_activemask

#include <cstdio>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────
// activemask 기본 — 분기에 따른 비트 패턴 변화
// ─────────────────────────────────────────────────────────────
__global__ void activeMaskBasics() {
    int lane = threadIdx.x % 32;
    if (lane == 0)
        printf("initial mask = 0x%08x\n", __activemask());

    if (threadIdx.x < 16) {
        if (lane == 0) printf("if mask    = 0x%08x\n", __activemask());
    } else {
        if (lane == 16) printf("else mask  = 0x%08x\n", __activemask());
    }
}

// ─────────────────────────────────────────────────────────────
// Race condition 방지: shared 초기화 후 syncwarp 필수
// ─────────────────────────────────────────────────────────────
__global__ void racy() {
    __shared__ int counter;
    if (threadIdx.x == 0) counter = 0;
    // ❌ syncwarp 누락 → counter가 아직 0이 아닐 수 있음
    if (threadIdx.x % 2 == 0) atomicAdd(&counter, 1);
    if (threadIdx.x == 31) printf("[racy] counter=%d\n", counter);
}

__global__ void safe() {
    __shared__ int counter;
    if (threadIdx.x == 0) counter = 0;
    __syncwarp(__activemask());         // ✅ 초기화 완료 보장
    if (threadIdx.x % 2 == 0) atomicAdd(&counter, 1);
    __syncwarp(__activemask());
    if (threadIdx.x == 31) printf("[safe] counter=%d\n", counter);
}

// ─────────────────────────────────────────────────────────────
// 마스크 비트 연산 — 활성 수, 첫/마지막 lane, 내 rank
// ─────────────────────────────────────────────────────────────
__global__ void maskOperations() {
    unsigned mask  = __activemask();
    int laneId     = threadIdx.x % 32;
    int activeCnt  = __popc(mask);
    int firstLane  = __ffs(mask) - 1;
    int lastLane   = 31 - __clz(mask);
    unsigned lower = mask & ((1U << laneId) - 1);
    int myRank     = __popc(lower);

    if (laneId == 0) {
        printf("active=%d first=%d last=%d (lane0 rank=%d)\n",
               activeCnt, firstLane, lastLane, myRank);
    }
}

// ─────────────────────────────────────────────────────────────
// 동적 마스크 관리 — 분기마다 부분 마스크로 sync
// ─────────────────────────────────────────────────────────────
__device__ void doWork1() { /* ... */ }
__device__ void doWork2() { /* ... */ }

__global__ void maskManagement() {
    unsigned originalMask = __activemask();

    if (threadIdx.x < 16) {
        unsigned partial = __activemask();
        doWork1();
        __syncwarp(partial);
    }
    __syncwarp(originalMask);

    if (threadIdx.x % 2 == 0) {
        unsigned even = __activemask();
        doWork2();
        __syncwarp(even);
    }
    __syncwarp(originalMask);
}

int main() {
    activeMaskBasics<<<1, 32>>>();
    racy<<<1, 32>>>();
    safe<<<1, 32>>>();
    maskOperations<<<1, 32>>>();
    maskManagement<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
