#include <cstdio>
#include <cuda_runtime.h>

constexpr int TPB = 8;   // 每 block 线程数，也是每 block 处理的数据量

__global__ void sum_kernel(float* out,const float* a,int size)
{
    __shared__ float cache[TPB];

    int block_start = blockIdx.x * blockDim.x;
    int i           = block_start + threadIdx.x;
    int local_i     = threadIdx.x;

    //global to shared
    if(i<size){
        cache[local_i] = a[i];
    }else{
        cache[local_i]=0.0f; //超过变成0,不影响求和
    }
    __syncthreads(); //搬运数据

    //求和
    for(int offset=1;offset<TPB;offset *= 2)
    {
        float val =0.0f;
        
        if(local_i >= offset) val = cache[local_i - offset];
        __syncthreads(); //同步

        cache[local_i] += val;
        __syncthreads();
    }

    //写出block sum,是一个block来求和
    if(local_i == TPB -1) out[blockIdx.x]=cache[local_i];
}

int main() {
    // ============ Test 1: Simple（SIZE = 8，1 个 block）============
    {
        const int SIZE = 8, BPG = 1;
        float *a, *out;
        cudaMallocManaged(&a,   SIZE * sizeof(float));
        cudaMallocManaged(&out, BPG  * sizeof(float));

        for (int i = 0; i < SIZE; ++i) a[i] = static_cast<float>(i);
        out[0] = 0.0f;

        sum_kernel<<<BPG, TPB>>>(out, a, SIZE);
        cudaDeviceSynchronize();

        printf("Test 1 (Simple): out[0] = %.1f\n", out[0]);
        cudaFree(a); cudaFree(out);
    }

    // ============ Test 2: Full（SIZE = 15，2 个 block）============
    {
        const int SIZE = 15;
        const int BPG  = (SIZE + TPB - 1) / TPB;   // ceil(15/8) = 2
        float *a, *out;
        cudaMallocManaged(&a,   SIZE * sizeof(float));
        cudaMallocManaged(&out, BPG  * sizeof(float));

        for (int i = 0; i < SIZE; ++i) a[i] = static_cast<float>(i);
        for (int j = 0; j < BPG; ++j)  out[j] = 0.0f;

        sum_kernel<<<BPG, TPB>>>(out, a, SIZE);
        cudaDeviceSynchronize();

        printf("Test 2 (Full):\n");
        for (int j = 0; j < BPG; ++j)
            printf("  out[%d] = %.1f\n", j, out[j]);
        cudaFree(a); cudaFree(out);
    }

    return 0;
}