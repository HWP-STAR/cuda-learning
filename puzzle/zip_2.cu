#include <cstdio>
#include <cuda_runtime.h>

constexpr int N = 4;   // 向量长度，也是要启动的线程数

__global__ void zip_add(int* out,const int* a,const int* b)
{
    int idx = threadIdx.x;        // 线程号即数据下标
    out[idx] = a[idx] + b[idx];
}

int main()
{
    int *a,*b,*out;
    cudaMallocManaged(&a,   N * sizeof(int));   // CPU/GPU 共享
    cudaMallocManaged(&b,   N * sizeof(int));
    cudaMallocManaged(&out, N * sizeof(int));

    for (int i = 0;i<N; ++i)
    {
        a[i] = i;
        b[i] = i;
    }
    zip_add<<<1,N>>>(out,a,b);                  // 1 block × N 线程
    cudaDeviceSynchronize();
    // 打印结果
    for (int i = 0; i < N; ++i)
        printf("out[%d] = %d\n", i, out[i]);

    // 释放
    cudaFree(a);
    cudaFree(b);
    cudaFree(out);
    return 0;
}