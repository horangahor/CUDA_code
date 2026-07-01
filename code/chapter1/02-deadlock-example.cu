// 02-deadlock-example.cu
// 슬라이드: part0/chapter1/04-sync-async-blocking.md
// __syncthreads()를 분기 안에서 호출하면 데드락 발생
//
// 빌드: nvcc 02-deadlock-example.cu -o deadlock_example
// 실행: ./deadlock_example
//
// ⚠️ "deadlock_kernel"은 실행 시 실제로 행이 걸리거나
//    cudaErrorLaunchTimeout으로 종료될 수 있습니다.
//    안전하게 보려면 SAFE 커널만 활성화하세요.

#include <cstdio>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────
// ❌ 위험: 일부 스레드만 __syncthreads()에 도달 → 데드락
// ─────────────────────────────────────────────────────────────
__global__ void deadlock_kernel(float* data, int condition) {
    int tid = threadIdx.x;
    if (tid < condition) {
        // 일부 스레드만 동기화 대기 — 나머지는 도달하지 않음
        __syncthreads();
    }
    data[tid] = (float)tid;
}

// ─────────────────────────────────────────────────────────────
// ✅ 안전: 분기 밖에서 __syncthreads()
// ─────────────────────────────────────────────────────────────
__global__ void safe_kernel(float* data, int condition) {
    int tid = threadIdx.x;
    float local = 0.0f;
    if (tid < condition) {
        local = (float)(tid * 2);
    }
    __syncthreads();              // 분기 밖 — 모든 스레드가 도달
    data[tid] = local;
}

int main() {
    constexpr int N = 256;
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    // SAFE 패턴 시연
    safe_kernel<<<1, N>>>(d_data, 128);
    cudaError_t err = cudaDeviceSynchronize();
    printf("safe_kernel: %s\n", cudaGetErrorString(err));

    // ⚠️ 데드락 시연 — 환경에 따라 hang/timeout 발생
    // deadlock_kernel<<<1, N>>>(d_data, 128);
    // err = cudaDeviceSynchronize();
    // printf("deadlock_kernel: %s\n", cudaGetErrorString(err));

    cudaFree(d_data);
    return 0;
}
