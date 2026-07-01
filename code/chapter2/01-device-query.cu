// 01-device-query.cu
// 슬라이드: part0/chapter2/02-compute-capability.md, 07-key-metrics.md
// cudaGetDeviceProperties()로 GPU 사양 조회
//
// 빌드: nvcc 01-device-query.cu -o device_query
// 실행: ./device_query

#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found\n");
        return 1;
    }

    for (int dev = 0; dev < device_count; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("=== Device %d ===\n", dev);
        printf("Name                  : %s\n", prop.name);
        printf("Compute Capability    : %d.%d\n", prop.major, prop.minor);
        printf("SM count              : %d\n", prop.multiProcessorCount);
        printf("Max threads / block   : %d\n", prop.maxThreadsPerBlock);
        printf("Max block dim         : (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dim          : (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("Warp size             : %d\n", prop.warpSize);
        printf("Total global memory   : %.2f GB\n",
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("Shared mem / block    : %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("L2 cache size         : %d KB\n", prop.l2CacheSize / 1024);
        printf("Memory clock rate     : %d MHz\n", prop.memoryClockRate / 1000);
        printf("Memory bus width      : %d bits\n", prop.memoryBusWidth);
        printf("Peak bandwidth (calc) : %.1f GB/s\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("Unified addressing    : %s\n", prop.unifiedAddressing ? "yes" : "no");
        printf("Concurrent kernels    : %s\n", prop.concurrentKernels ? "yes" : "no");
    }
    return 0;
}
