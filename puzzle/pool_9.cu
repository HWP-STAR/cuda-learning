#include <cstdio>
#include <cuda_runtime.h>

constexpr int N   = 8;   // 数据长度
constexpr int TPB = 8;   // 每 block 线程数
constexpr int BPG = 1;   // block 数 = ceil(8/8) = 1

__global__ void pool_kernel(float* out,const float* a,int size)
{
    __shared__ float shared[TPB];
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int local_i= threadIdx.x;

    if(i<size) shared[local_i] = a[i];
    __syncthreads();

    if(i< size)
    {
        float acc = shared[local_i];
        if(local_i >=1) acc += shared[local_i -1];
        if(local_i >=2) acc += shared[local_i -2];
        out[i] = acc;
    }
}

int main() {
    float *a, *out;
    cudaMallocManaged(&a,   N * sizeof(float));
    cudaMallocManaged(&out, N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        a[i]   = static_cast<float>(i);   // a = [0, 1, 2, 3, 4, 5, 6, 7]
        out[i] = 0.0f;
    }

    pool_kernel<<<BPG,TPB>>>(out,a,N);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; ++i)
        printf("out[%d] = %.1f\n", i, out[i]);

    cudaFree(a);
    cudaFree(out);
    return 0;
}