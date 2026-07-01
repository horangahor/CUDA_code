// 02-host-vs-device.cu
// 슬라이드: part1/chapter3/02-host-vs-device.md
// h_/d_ 네이밍, 비동기 실행 모델, CPU/GPU 동시 작업
//
// 빌드: nvcc 02-host-vs-device.cu -o host_vs_device
// 실행: ./host_vs_device

#include <cstdio>
#include <cuda_runtime.h>

__global__ void scale_kernel(float* d_data, int n, float k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_data[i] *= k;
}

void preprocessing(float* h_data, int n) {
    for (int i = 0; i < n; ++i) h_data[i] = (float)i;
}

void do_other_work() {
    // GPU가 실행 중일 때 CPU가 처리할 수 있는 독립 작업
    volatile double sum = 0.0;
    for (int i = 0; i < 1'000'000; ++i) sum += (double)i;
    (void)sum;
}

int main() {
    constexpr int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    // Host 메모리 (h_ 접두사)
    float* h_data = (float*)malloc(bytes);
    preprocessing(h_data, N);

    // Device 메모리 (d_ 접두사)
    float* d_data;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    // 커널 실행 — 비동기로 즉시 리턴
    int block = 256;
    int grid  = (N + block - 1) / block;
    scale_kernel<<<grid, block>>>(d_data, N, 2.5f);

    // GPU 실행 중 CPU는 다른 작업 가능
    do_other_work();

    // GPU 완료 대기 후 결과 회수
    cudaDeviceSynchronize();
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

    printf("h_data[0]=%.2f, h_data[N-1]=%.2f\n", h_data[0], h_data[N - 1]);

    cudaFree(d_data);
    free(h_data);
    return 0;
}
