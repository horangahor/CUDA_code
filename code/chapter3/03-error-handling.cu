// 03-error-handling.cu
// 슬라이드: part1/chapter3/03-error-handling.md
// CUDA_CHECK 매크로, sync vs async 에러, sticky error 격리
//
// 빌드: nvcc 03-error-handling.cu -o error_handling
// 실행: ./error_handling

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────
// CUDA_CHECK 매크로 — 에러 발생 시 파일/라인 출력 후 종료
// ─────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

__global__ void simple_kernel(float* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = (float)i;
}

int main() {
    constexpr int N = 1024;
    size_t bytes = N * sizeof(float);
    float* d_data = nullptr;

    // 1. 정상 패턴 — 모든 호출에 CUDA_CHECK
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    int block = 256;
    int grid  = (N + block - 1) / block;
    simple_kernel<<<grid, block>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());        // 실행 구성 에러
    CUDA_CHECK(cudaDeviceSynchronize());   // 실행 중 에러

    // 2. 에러 격리 — 각 커널마다 sync로 sticky error 분리
    simple_kernel<<<grid, block>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());   // 첫 번째 커널 에러 확인
    simple_kernel<<<grid, block>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());   // 두 번째 커널 에러 확인

    // 3. 의도적 에러: 블록 크기 초과 (cudaErrorInvalidConfiguration)
    simple_kernel<<<1, 2048>>>(d_data, N);
    cudaError_t err = cudaGetLastError();
    printf("Expected error: %s\n", cudaGetErrorString(err));

    CUDA_CHECK(cudaFree(d_data));
    return 0;
}
