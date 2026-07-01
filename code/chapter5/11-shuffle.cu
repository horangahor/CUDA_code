// 11-shuffle.cu
// 슬라이드: part1/chapter5/11-shuffle-operations.md
// Shuffle 4종 + warpReduceSum + warpInclusiveScan + Bitonic + 매트릭스-벡터
//
// 빌드: nvcc -arch=sm_87 -O3 11-shuffle.cu -o shuffle_ops  # Jetson Orin Nano
// 실행: ./shuffle_ops

#include <cstdio>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────
// 4가지 shuffle 패턴 시연
// ─────────────────────────────────────────────────────────────
__global__ void allShufflePatterns() {
    int tid       = threadIdx.x;
    unsigned mask = __activemask();
    int value     = tid * 10;

    int from_5    = __shfl_sync     (mask, value, 5);     // broadcast
    int from_up   = __shfl_up_sync  (mask, value, 2);     // scan용
    int from_down = __shfl_down_sync(mask, value, 2);     // reduction용
    int from_xor  = __shfl_xor_sync (mask, value, 7);     // butterfly

    if (tid == 6) {
        printf("lane6: from_5=%d up2=%d down2=%d xor7=%d\n",
               from_5, from_up, from_down, from_xor);
    }
}

// ─────────────────────────────────────────────────────────────
// warpReduceSum — tree reduction (lane 0에 결과)
// ─────────────────────────────────────────────────────────────
__device__ float warpReduceSum(float v) {
    unsigned mask = __activemask();
    for (int off = 16; off > 0; off /= 2) {
        v += __shfl_down_sync(mask, v, off);
    }
    return v;
}

// ─────────────────────────────────────────────────────────────
// warpInclusiveScan — Kogge-Stone (각 lane이 자신까지의 누적합)
// ─────────────────────────────────────────────────────────────
__device__ float warpInclusiveScan(float v) {
    unsigned mask = __activemask();
    for (int off = 1; off < 32; off *= 2) {
        float t = __shfl_up_sync(mask, v, off);
        if ((threadIdx.x % 32) >= off) v += t;
    }
    return v;
}

// ─────────────────────────────────────────────────────────────
// Bitonic Sort (32 원소, lane 단위)
// ─────────────────────────────────────────────────────────────
__device__ void warpBitonicSort(int* val_per_lane) {
    int laneId    = threadIdx.x % 32;
    unsigned mask = __activemask();
    int v         = val_per_lane[laneId];

    for (int k = 2; k <= 32; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int partner = laneId ^ j;
            int other   = __shfl_sync(mask, v, partner);
            bool asc        = ((laneId & k) == 0);
            bool shouldSwap = (asc && v > other) || (!asc && v < other);
            if (shouldSwap) v = other;
        }
    }
    val_per_lane[laneId] = v;
}

// ─────────────────────────────────────────────────────────────
// 매트릭스-벡터 곱 — 워프당 한 행, shuffle reduction
// ─────────────────────────────────────────────────────────────
__global__ void matrixVectorMul(const float* M, const float* x,
                                float* y, int n) {
    int row = blockIdx.x;
    float dot = 0.0f;
    for (int col = threadIdx.x; col < n; col += 32) {
        dot += M[row * n + col] * x[col];
    }
    dot = warpReduceSum(dot);
    if ((threadIdx.x % 32) == 0) y[row] = dot;
}

// ─────────────────────────────────────────────────────────────
// Shuffle 검증 — 왕복 결과는 원본과 동일해야 함
// ─────────────────────────────────────────────────────────────
__global__ void verifyShuffleOperations() {
    int tid       = threadIdx.x;
    unsigned mask = __activemask();
    int original  = tid * 100;

    int reversed  = __shfl_sync(mask, original, 31 - (tid % 32));
    int restored  = __shfl_sync(mask, reversed, 31 - (tid % 32));
    if (original != restored)
        printf("[reverse] mismatch at lane %d\n", tid);

    int xored = __shfl_xor_sync(mask, original, 5);
    int back  = __shfl_xor_sync(mask, xored,    5);
    if (original != back)
        printf("[xor] mismatch at lane %d\n", tid);

    float v   = 1.0f;
    float sum = warpReduceSum(v);
    if (tid == 0) {
        int active = __popc(mask);
        printf("warpReduceSum(1) = %.0f (expected %d)\n", sum, active);
    }
}

// ─────────────────────────────────────────────────────────────
// Bitonic 호스트 launcher
// ─────────────────────────────────────────────────────────────
__global__ void bitonic_launcher(const int* in, int* out) {
    __shared__ int buf[32];
    if (threadIdx.x < 32) buf[threadIdx.x] = in[threadIdx.x];
    __syncthreads();
    if (threadIdx.x < 32) warpBitonicSort(buf);
    __syncthreads();
    if (threadIdx.x < 32) out[threadIdx.x] = buf[threadIdx.x];
}

int main() {
    allShufflePatterns<<<1, 32>>>();
    verifyShuffleOperations<<<1, 32>>>();

    constexpr int N = 32;
    int h_in[N], h_out[N];
    for (int i = 0; i < N; ++i) h_in[i] = N - i;       // 역순 입력

    int *d_in, *d_out;
    cudaMalloc(&d_in,  sizeof(h_in));
    cudaMalloc(&d_out, sizeof(h_out));
    cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice);
    bitonic_launcher<<<1, 32>>>(d_in, d_out);
    cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);

    printf("bitonic sorted (asc): ");
    for (int i = 0; i < N; ++i) printf("%d ", h_out[i]);
    printf("\n");

    constexpr int M = 64;
    float *d_M, *d_x, *d_y;
    cudaMalloc(&d_M, M * M * sizeof(float));
    cudaMalloc(&d_x, M * sizeof(float));
    cudaMalloc(&d_y, M * sizeof(float));
    matrixVectorMul<<<M, 32>>>(d_M, d_x, d_y, M);

    cudaDeviceSynchronize();
    cudaFree(d_in); cudaFree(d_out);
    cudaFree(d_M); cudaFree(d_x); cudaFree(d_y);
    return 0;
}
