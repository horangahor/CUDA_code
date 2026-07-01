// 01-simt-vs-mimd.cu
// 슬라이드: part0/chapter1/03-simt-model.md
// CPU MIMD 분기 vs GPU SIMT 마스킹 비교
//
// 빌드: nvcc 01-simt-vs-mimd.cu -o simt_vs_mimd
// 실행: ./simt_vs_mimd

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────
// CPU 버전: MIMD (Multiple Instruction, Multiple Data)
//   각 코어가 독립적으로 분기를 평가
// ─────────────────────────────────────────────────────────────
void cpu_mimd(const float* in, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        if (in[i] > 0.0f)
            out[i] = sqrtf(in[i]);
        else
            out[i] = 0.0f;
    }
}

// ─────────────────────────────────────────────────────────────
// GPU 버전: SIMT (Single Instruction, Multiple Threads)
//   32개 스레드가 같은 명령어 실행, 분기는 마스킹으로 처리
// ─────────────────────────────────────────────────────────────
__global__ void gpu_simt(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // 모든 스레드가 동일 instruction stream — 조건은 predicate로 처리
    out[idx] = (in[idx] > 0.0f) ? sqrtf(in[idx]) : 0.0f;
}

int main() {
    constexpr int N = 1 << 16;
    size_t bytes = N * sizeof(float);

    float *h_in = (float*)malloc(bytes);
    float *h_out_cpu = (float*)malloc(bytes);
    float *h_out_gpu = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i)
        h_in[i] = (i % 2 == 0) ? (float)i : -(float)i;

    cpu_mimd(h_in, h_out_cpu, N);

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (N + block - 1) / block;
    gpu_simt<<<grid, block>>>(d_in, d_out, N);
    cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost);

    int mismatches = 0;
    for (int i = 0; i < N; ++i)
        if (fabsf(h_out_cpu[i] - h_out_gpu[i]) > 1e-5f) mismatches++;
    printf("CPU vs GPU mismatches: %d / %d\n", mismatches, N);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out_cpu); free(h_out_gpu);
    return 0;
}
