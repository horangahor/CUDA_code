// 03-independent-thread-scheduling.cu
// 슬라이드: part1/chapter5/03-independent-thread-scheduling.md
// Volta+ Independent Thread Scheduling — 독립 PC + 명시적 __syncwarp
//
// 빌드: nvcc -arch=sm_87 -O3 03-independent-thread-scheduling.cu -o its_demo  # Jetson Orin Nano
// 실행: ./its_demo

#include <cstdio>
#include <cuda_runtime.h>

__device__ void doWork()      { /* ... */ }
__device__ void doOtherWork() { /* ... */ }

// ─────────────────────────────────────────────────────────────
// Pre-Volta 스타일: 자동 수렴 가정 (Volta+에서는 race 발생 가능)
// ─────────────────────────────────────────────────────────────
__global__ void oldModel() {
    if (threadIdx.x < 16) {
        doWork();           // T0-T15
    } else {
        doOtherWork();      // T16-T31
    }
    // 이전엔 자동 재수렴 — Volta+에서는 보장되지 않음
}

// ─────────────────────────────────────────────────────────────
// Volta+ 스타일: 명시적 __syncwarp 사용
// ─────────────────────────────────────────────────────────────
__global__ void newModel() {
    unsigned mask = __activemask();
    if (threadIdx.x < 16) {
        doWork();
    } else {
        doOtherWork();
    }
    __syncwarp(mask);       // 명시적 수렴 — 분기 후 안전한 공통 코드
}

// ─────────────────────────────────────────────────────────────
// Race condition 시연: shared memory 쓰기 후 동기화 누락
// ─────────────────────────────────────────────────────────────
__global__ void unsafeShared() {
    __shared__ int data[32];
    if (threadIdx.x == 0) data[0] = 42;
    int v = data[0];                    // ❌ Race condition
    if (threadIdx.x == 31 && v != 42) {
        printf("[unsafe] read got %d (race)\n", v);
    }
}

__global__ void safeShared() {
    __shared__ int data[32];
    unsigned mask = __activemask();
    if (threadIdx.x == 0) data[0] = 42;
    __syncwarp(mask);                   // ✅ 쓰기 후 동기화
    int v = data[0];
    if (threadIdx.x == 31 && v != 42) {
        printf("[safe] read got %d\n", v);
    }
}

// ─────────────────────────────────────────────────────────────
// 호환성: 컴파일 타임 / 런타임 체크
// ─────────────────────────────────────────────────────────────
__global__ void portableCode() {
#if __CUDA_ARCH__ >= 700
    unsigned mask = __activemask();
    doWork();
    __syncwarp(mask);
#else
    doWork();
#endif
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Compute Capability: %d.%d (ITS supported: %s)\n",
           prop.major, prop.minor,
           (prop.major >= 7) ? "yes" : "no");

    oldModel<<<1, 32>>>();
    newModel<<<1, 32>>>();
    unsafeShared<<<1, 32>>>();
    safeShared<<<1, 32>>>();
    portableCode<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
