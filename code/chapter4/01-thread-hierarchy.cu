// 01-thread-hierarchy.cu
// 슬라이드: part1/chapter4/01-thread-hierarchy.md
// Grid → Block → Thread 3계층, __syncthreads() 동작 확인
//
// 빌드: nvcc 01-thread-hierarchy.cu -o thread_hierarchy
// 실행: ./thread_hierarchy

#include <cstdio>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────
// Thread 단위 — 각 스레드는 독립 레지스터 + threadIdx
// ─────────────────────────────────────────────────────────────
__global__ void thread_demo() {
    int tid = threadIdx.x;
    int local_var = tid * 2;            // 스레드별 레지스터
    if (tid == 0) {
        printf("Block %d: first thread, local_var=%d\n",
               blockIdx.x, local_var);
    }
}

// ─────────────────────────────────────────────────────────────
// Block 단위 — __syncthreads()로 블록 내 동기화 가능
// ─────────────────────────────────────────────────────────────
__global__ void block_demo(int* out) {
    __shared__ int shared_buf[256];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    shared_buf[tid] = bid * blockDim.x + tid;
    __syncthreads();                    // 블록 내 동기화 ✅

    int neighbor = shared_buf[(tid + 1) % blockDim.x];
    out[bid * blockDim.x + tid] = neighbor;
}

// ─────────────────────────────────────────────────────────────
// Grid 단위 — 모든 스레드가 동일 커널, blockIdx로 위치 식별
// ─────────────────────────────────────────────────────────────
__global__ void grid_demo(int* data, int N) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < N) data[global_id] = global_id;
}

int main() {
    thread_demo<<<2, 4>>>();
    cudaDeviceSynchronize();

    constexpr int N = 1024;
    int *d_out, *d_data;
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMalloc(&d_data, N * sizeof(int));

    block_demo<<<4, 256>>>(d_out);
    grid_demo<<<(N + 255) / 256, 256>>>(d_data, N);
    cudaDeviceSynchronize();

    int h_data[8];
    cudaMemcpy(h_data, d_data, sizeof(h_data), cudaMemcpyDeviceToHost);
    printf("data[0..7] = ");
    for (int i = 0; i < 8; ++i) printf("%d ", h_data[i]);
    printf("\n");

    cudaFree(d_out); cudaFree(d_data);
    return 0;
}
