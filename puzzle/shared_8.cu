#include <cstdio>
#include <cuda_runtime.h>

constexpr int N   = 8;   // 数据长度
constexpr int TPB = 4;   // 每 block 线程数
constexpr int BPG = 2;   // block 数 = ceil(8/4) = 2

__global__ void shared_demo(float* out,const float* a,int size)
{
    __shared__ float shared[TPB];
    int i       = blockIdx.x * blockDim.x + threadIdx.x;  // 全局下标 0..7
    int local_i = threadIdx.x;                             // 块内下标 0..3

    // 全局to共享
    if(i < size) shared[local_i] = a[i];
    __syncthreads();

    //共享to全局
    if(i<size) out[i] = shared[local_i] + 10.0f;
}

int main() {
    float *a, *out;
    cudaMallocManaged(&a,   N * sizeof(float));
    cudaMallocManaged(&out, N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        a[i]   = 1.0f;   // a 全 1
        out[i] = 0.0f;
    }
    shared_demo<<<BPG,TPB>>>(out,a,N);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; ++i)
        printf("out[%d] = %.1f\n", i, out[i]);

    cudaFree(a);
    cudaFree(out);
    return 0;
}