#include <cstdio>
#include <cuda_runtime.h>

constexpr int N        = 15;   // 数据长度
constexpr int TPB      = 4;   // 每个 block 的线程数（Threads Per Block）
constexpr int NUM_BLKS = 4;   // block 数 = ceil(N / TPB) = ceil(9/4) = 3

__global__ void map_block(float* out, const float* a,int size)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<size)
        out[i]=a[i] +10.0f;
}

int main() {
    float *a, *out;
    cudaMallocManaged(&a,   N * sizeof(float));
    cudaMallocManaged(&out, N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        a[i]   = static_cast<float>(i);   // a = [0, 1, ..., 8]
        out[i] = 0.0f;
    }

    map_block<<<NUM_BLKS,TPB>>>(out,a,N);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; ++i)
        printf("out[%d] = %.1f\n", i, out[i]);

    cudaFree(a);
    cudaFree(out);
    return 0;
}