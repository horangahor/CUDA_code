// 04-instruction-issue.cu
// 슬라이드: part1/chapter5/04-instruction-issue.md
// Dual issue (FP+INT 교차), Scoreboard / RAW hazard 패턴
//
// 빌드: nvcc -O3 04-instruction-issue.cu -o instruction_issue
// 실행: ./instruction_issue

#include <cstdio>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────
// Dual issue 친화적: FP32 + INT32 교차 배치
// ─────────────────────────────────────────────────────────────
__global__ void dualIssueKernel(const float* data, const float* coeffs,
                                float* output, int stride) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    int   offset = 0;
    #pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        // FP32 유닛 + INT32 유닛이 동시에 발행 가능
        float val = data[idx + offset];
        result += val * coeffs[i];
        offset += stride;
    }
    output[idx] = result;
}

// ─────────────────────────────────────────────────────────────
// 안티패턴: SFU 연속 사용 (sin/cos/sqrt/exp) → 한 유닛 포화
// ─────────────────────────────────────────────────────────────
__global__ void sfuHeavy(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    out[idx] = sinf(v) + cosf(v) + sqrtf(v) + expf(v);
}

// ─────────────────────────────────────────────────────────────
// RAW Hazard — Scoreboard stall 시연
// ─────────────────────────────────────────────────────────────
__global__ void rawHazard_bad(const float* data, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float a = data[idx];        // LD (~400 cycles)
    float b = a * 2.0f;         // a 의존 → STALL!
    out[idx] = b;
}

__global__ void rawHazard_good(const float* data, const float* other,
                               float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float a = data[idx];        // LD a
    float x = other[idx];       // LD x — a와 독립
    float y = x + 1.0f;         // x 사용 — 이 동안 a 도착
    float b = a * 2.0f;         // 도착 시점 → stall 없음
    out[idx] = b + y;
}

int main() {
    constexpr int N = 1 << 20;
    float *d_data, *d_other, *d_out;
    float h_coeffs[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    float* d_coeffs;
    cudaMalloc(&d_data,   N * sizeof(float));
    cudaMalloc(&d_other,  N * sizeof(float));
    cudaMalloc(&d_out,    N * sizeof(float));
    cudaMalloc(&d_coeffs, 8 * sizeof(float));
    cudaMemcpy(d_coeffs, h_coeffs, sizeof(h_coeffs), cudaMemcpyHostToDevice);

    int block = 256;
    int grid  = (N + block - 1) / block;

    dualIssueKernel<<<grid, block>>>(d_data, d_coeffs, d_out, 1);
    sfuHeavy<<<grid, block>>>(d_data, d_out, N);
    rawHazard_bad<<<grid, block>>>(d_data, d_out, N);
    rawHazard_good<<<grid, block>>>(d_data, d_other, d_out, N);
    cudaDeviceSynchronize();

    cudaFree(d_data); cudaFree(d_other); cudaFree(d_out); cudaFree(d_coeffs);
    return 0;
}
