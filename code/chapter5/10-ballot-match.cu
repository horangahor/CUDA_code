// 10-ballot-match.cu
// 슬라이드: part1/chapter5/10-ballot-match-intrinsics.md
// Ballot 기반 stream compaction, Match 기반 그룹화 (해시 dedup)
//
// 빌드: nvcc -arch=sm_87 -O3 10-ballot-match.cu -o ballot_match  # Jetson Orin Nano
// 실행: ./ballot_match

#include <cstdio>
#include <cuda_runtime.h>

__device__ void   allWork()         { /* ... */ }
__device__ void   partialWork()     { /* ... */ }
__device__ int    hashLookup(int k) { return k * 1234567 + 1; }

// ─────────────────────────────────────────────────────────────
// Ballot 기반 stream compaction (워프 단위)
//   조건 만족하는 원소만 출력 배열 앞쪽에 빈틈 없이 저장
// ─────────────────────────────────────────────────────────────
__global__ void compactWithBallot(const float* in, float* out, int* counter, int n) {
    int tid       = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    unsigned mask  = __activemask();
    float    value = in[tid];
    bool     keep  = (value > 0.0f);

    unsigned ballot = __ballot_sync(mask, keep);

    if (keep) {
        // 내 lane 미만에서 keep된 개수 = 워프 내 출력 위치
        unsigned lower = ballot & ((1U << (threadIdx.x % 32)) - 1);
        int local_idx  = __popc(lower);

        // 워프당 한 번만 글로벌 카운터 갱신
        int leader = __ffs(ballot) - 1;
        int warp_count = __popc(ballot);
        int warp_base;
        if ((threadIdx.x % 32) == leader) {
            warp_base = atomicAdd(counter, warp_count);
        }
        warp_base = __shfl_sync(mask, warp_base, leader);

        out[warp_base + local_idx] = value;
    }
}

// ─────────────────────────────────────────────────────────────
// 조기 종료 / 전체 경로 / 부분 경로 분기
// ─────────────────────────────────────────────────────────────
__global__ void path_select(const int* needs, int n) {
    int tid       = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    unsigned mask     = __activemask();
    bool     needsWork = (needs[tid] != 0);
    unsigned ballot   = __ballot_sync(mask, needsWork);

    if (ballot == 0)        return;     // 전체 종료
    if (ballot == mask)   { allWork();  return; }
    if (needsWork)          partialWork();
}

// ─────────────────────────────────────────────────────────────
// Match 기반 해시 dedup — 같은 키 검색은 leader만 수행
// ─────────────────────────────────────────────────────────────
__global__ void hashDedup(const int* queries, int* output, int n) {
    int tid           = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    unsigned mask     = __activemask();
    int      key      = queries[tid];
    unsigned same     = __match_any_sync(mask, key);
    int      leader   = __ffs(same) - 1;
    int      myLane   = threadIdx.x % 32;

    int result;
    if (myLane == leader) {
        result = hashLookup(key);                   // leader만 검색
    }
    result = __shfl_sync(mask, result, leader);     // 그룹 내 공유
    output[tid] = result;
}

int main() {
    constexpr int N = 1024;
    float *d_in, *d_out;
    int   *d_counter, *d_needs, *d_q, *d_o;
    cudaMalloc(&d_in,      N * sizeof(float));
    cudaMalloc(&d_out,     N * sizeof(float));
    cudaMalloc(&d_counter, sizeof(int));
    cudaMalloc(&d_needs,   N * sizeof(int));
    cudaMalloc(&d_q,       N * sizeof(int));
    cudaMalloc(&d_o,       N * sizeof(int));

    int zero = 0;
    cudaMemcpy(d_counter, &zero, sizeof(int), cudaMemcpyHostToDevice);

    int block = 256;
    int grid  = (N + block - 1) / block;

    compactWithBallot<<<grid, block>>>(d_in, d_out, d_counter, N);
    path_select<<<grid, block>>>(d_needs, N);
    hashDedup<<<grid, block>>>(d_q, d_o, N);
    cudaDeviceSynchronize();

    int kept = 0;
    cudaMemcpy(&kept, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("compaction kept = %d / %d\n", kept, N);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_counter);
    cudaFree(d_needs); cudaFree(d_q); cudaFree(d_o);
    return 0;
}
