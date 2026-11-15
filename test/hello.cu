#include "cuda_runtime.h"
#include <stdio.h>

#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (code %d)\n", cudaGetErrorString(err), err); \
            return 1; \
        } \
    } while(0)

__global__ void hello()
{
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main()
{
    hello<<<1, 1>>>(); // 内核启动本身不返回错误，需要后续检查
    CHECK( cudaGetLastError() );      // 检查内核启动是否失败
    CHECK( cudaDeviceSynchronize() ); // 检查内核执行是否出错
    return 0;
}