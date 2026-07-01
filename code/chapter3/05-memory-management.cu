// 05-memory-management.cu
// 슬라이드: part1/chapter3/05-memory-management.md
// Allocate → Copy → Compute → Copy → Free 패턴 + 흔한 실수 시연
//
// 빌드: nvcc 05-memory-management.cu -o memory_management
// 실행: ./memory_management

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void touch_kernel(float* d_data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) d_data[i] = (float)i;
}

// ─────────────────────────────────────────────────────────────
// 안전한 메모리 관리 패턴
// ─────────────────────────────────────────────────────────────
void safe_pattern() {
    constexpr int N = 1024;
    size_t size = N * sizeof(float);
    float* h_data = (float*)malloc(size);

    float* d_data;
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    touch_kernel<<<(N + 255) / 256, 256>>>(d_data, N);
    cudaDeviceSynchronize();                              // 커널 완료 보장
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    d_data = nullptr;                                     // 댕글링 방지
    free(h_data);
}

// ─────────────────────────────────────────────────────────────
// ❌ 흔한 실수 1: 복사 방향 오류
// ─────────────────────────────────────────────────────────────
void mistake_copy_direction() {
    constexpr int N = 16;
    size_t size = N * sizeof(float);
    float* h_data = (float*)malloc(size);
    float* d_data;
    cudaMalloc(&d_data, size);

    // ❌ Device→Host인데 cudaMemcpyHostToDevice 플래그
    cudaMemcpy(h_data, d_data, size, cudaMemcpyHostToDevice);
    cudaError_t err = cudaGetLastError();
    printf("[mistake] copy direction: %s\n", cudaGetErrorString(err));

    // ✅ 올바른 방향
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    free(h_data);
}

// ─────────────────────────────────────────────────────────────
// ❌ 흔한 실수 2: 타입 크기 불일치
// ─────────────────────────────────────────────────────────────
void mistake_type_mismatch() {
    constexpr int N = 100;
    int*   h_int = (int*)malloc(N * sizeof(int));
    float* d_float;
    cudaMalloc(&d_float, N * sizeof(float));

    // ❌ int 배열을 float 버퍼에 복사 (sizeof는 같지만 의미 오류)
    cudaMemcpy(d_float, h_int, N * sizeof(int), cudaMemcpyHostToDevice);

    // ✅ 같은 타입으로 통일
    int* d_int;
    cudaMalloc(&d_int, N * sizeof(int));
    cudaMemcpy(d_int, h_int, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaFree(d_int); cudaFree(d_float);
    free(h_int);
}

int main() {
    safe_pattern();
    mistake_copy_direction();
    mistake_type_mismatch();
    printf("done\n");
    return 0;
}
