// 02-warp-scheduling.cu
// 슬라이드: part1/chapter5/02-warp-scheduling.md
// Stall 최소화 패턴, launch_bounds, warp specialization (Hopper+)
//
// 빌드: nvcc -O3 02-warp-scheduling.cu -o warp_scheduling
// 실행: ./warp_scheduling

#include <cstdio>
#include <cuda_runtime.h>

__device__ float compute1(float x) { return x * 1.1f + 0.5f; }
__device__ float compute2(float x) { return x * 0.9f - 0.3f; }

// ─────────────────────────────────────────────────────────────
// Stall 최소화: 메모리 로드를 모아 발행 → 도착 대기 동안 독립 연산
// ─────────────────────────────────────────────────────────────
__global__ void optimized_schedule(const float* g, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx + 1 >= n) return;

    // 1. 메모리 접근 모음 (동시 in-flight)
    float a = g[idx];
    float b = g[idx + 1];

    // 2. 대기 중 독립 연산 (a, b와 무관)
    float x = compute1((float)idx);
    float y = compute2((float)idx);

    // 3. a, b 도착 시점에 사용 → stall 없음
    out[idx]     = a * x;
    out[idx + 1] = b * y;
}

// ─────────────────────────────────────────────────────────────
// launch_bounds — 컴파일러에게 occupancy 힌트 제공
//   "블록당 최대 256, SM당 최소 4블록 실행 보장하려 노력"
// ─────────────────────────────────────────────────────────────
__global__ void __launch_bounds__(256, 4) hinted_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float v = data[idx];
    #pragma unroll 8
    for (int i = 0; i < 32; ++i) v = v * 1.001f + 0.001f;
    data[idx] = v;
}

// ─────────────────────────────────────────────────────────────
// Warp Specialization (Hopper+) — Producer/Consumer 분리
//   블록 내 일부 워프는 데이터 생산, 나머지는 소비
// ─────────────────────────────────────────────────────────────
__device__ volatile int g_done = 0;
__device__ float g_buffer[256];

__device__ void produceData(int lane)  { g_buffer[lane] = (float)lane; }
__device__ void consumeData(int lane)  { (void)g_buffer[lane]; }

__global__ void warpSpecialization() {
    int warp = threadIdx.x / 32;
    if (warp < 4) {                                   // Producer warps
        produceData(threadIdx.x % 32);
        __threadfence_block();                        // 가시성 보장
    } else {                                          // Consumer warps
        consumeData(threadIdx.x % 32);
    }
}

int main() {
    constexpr int N = 1 << 16;
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    int block = 256;
    int grid  = (N + block - 1) / block;

    optimized_schedule<<<grid, block>>>(d_in, d_out, N);
    hinted_kernel<<<grid, 256>>>(d_in, N);
    warpSpecialization<<<1, 256>>>();
    cudaDeviceSynchronize();

    printf("warp_scheduling demo done\n");
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
