// map.cu
#include <cstdio>
#include <cuda_runtime.h>

constexpr int N = 4;

// kernel：一个线程处理一个元素
__global__ void add_ten(int* out, const int* a) {
    int idx = threadIdx.x;        // 线程号即数据下标
    out[idx] = a[idx] + 10;
}

int main() {
    int *a, *out;
    cudaMallocManaged(&a,   N * sizeof(int));   // CPU/GPU 共享
    cudaMallocManaged(&out, N * sizeof(int));

    for (int i = 0; i < N; ++i) a[i] = i;       // a = {0,1,2,3}

    add_ten<<<1, N>>>(out, a);                  // 1 block × N 线程
    cudaDeviceSynchronize();

    for (int i = 0; i < N; ++i)
        printf("out[%d] = %d\n", i, out[i]);

    cudaFree(a);
    cudaFree(out);
}