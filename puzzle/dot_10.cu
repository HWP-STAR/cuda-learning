#include <cstdio>
#include <cuda_runtime.h>

constexpr int N   = 8;   // 向量长度
constexpr int TPB = 8;   // 每 block 线程数（= N，恰好 1 线程 1 元素）

__global__ void dot_kernel(float* out,const float* a, const float* b,int size)
{
    __shared__ float shared[TPB];
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int local_i=threadIdx.x;

    //全局算*
    if(i<size) shared[local_i] = a[i] * b[i];
    else shared[local_i] = 0.0f;

    __syncthreads();

    //共享内存内归约
    for(int stride= TPB/2;stride > 0;stride >>= 1) //右位移，就是/2
    {
        if(local_i < stride) shared[local_i] +=shared[local_i+stride];
        __syncthreads();
    }

    //写会全局
    if(local_i == 0) out[0] = shared[0];
}

int main() {
    float *a, *b, *out;
    cudaMallocManaged(&a,   N * sizeof(float));
    cudaMallocManaged(&b,   N * sizeof(float));
    cudaMallocManaged(&out, 1 * sizeof(float));   // 输出是标量

    for (int i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i);   // a = [0, 1, ..., 7]
        b[i] = static_cast<float>(i);   // b = [0, 1, ..., 7]
    }
    out[0] = 0.0f;

    dot_kernel<<<1,TPB>>>(out,a,b,N);
    cudaDeviceSynchronize();

    printf("out[0] = %.1f\n", out[0]);

    cudaFree(a);
    cudaFree(b);
    cudaFree(out);
    return 0;
}
