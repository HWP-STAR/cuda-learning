#include <cstdio>
#include <cuda_runtime.h>

constexpr int TPB         = 8;   // 每 block 线程数
constexpr int MAX_CONV    = 4;   // 卷积核最大长度（编译期常量，决定共享内存大小）
constexpr int TPB_MAX_CONV = TPB + MAX_CONV;   // 12：本 block 的 TPB 个 + 最多 MAX_CONV-1 个 halo

__global__ void conv_kernel(float* out,const float* a,const float* b,int a_size,int b_size)
{
    __shared__ float shared_a[TPB_MAX_CONV]; //block + halo
    __shared__ float shared_b[MAX_CONV]; //卷积kernel

    int block_start = blockIdx.x*blockDim.x;
    int i = block_start + threadIdx.x;//全局下标
    int local_i = threadIdx.x;

    //to共享内存
    if(i < a_size){
        shared_a[local_i] = a[i];
    }else{
        shared_a[local_i] = 0.0f;
    }

    if(local_i < b_size){
        shared_b[local_i] = b[local_i];
    }

    // (3) halo 加载：需要 b_size - 1 个额外元素
    if(local_i >= b_size && local_i < b_size+b_size-1)
    {
        int halo_idx = local_i - b_size;
        int global_idx = block_start + TPB + halo_idx;
        if(global_idx < a_size){
            shared_a[TPB + halo_idx] = a[global_idx];
        }else{
            shared_a[TPB+ halo_idx] = 0.0f; //越界
        }
    }
    __syncthreads();

    //计算卷积
    if(i < a_size)
    {
        float acc = 0.0f;
        for(int j=0;j<b_size;++j)
        {
            acc += shared_a[local_i + j] * shared_b[j];
        }
        out[i] = acc;
    }
}

int main() {
    // ============ Test 1: Simple ============
    {
        const int A_SIZE = 6, B_SIZE = 3, BPG = 1;
        float *a, *b, *out;
        cudaMallocManaged(&a,   A_SIZE * sizeof(float));
        cudaMallocManaged(&b,   B_SIZE * sizeof(float));
        cudaMallocManaged(&out, A_SIZE * sizeof(float));

        for (int i = 0; i < A_SIZE; ++i) a[i] = static_cast<float>(i);
        for (int j = 0; j < B_SIZE; ++j) b[j] = static_cast<float>(j);
        for (int i = 0; i < A_SIZE; ++i) out[i] = 0.0f;

        conv_kernel<<<BPG, TPB>>>(out, a, b, A_SIZE, B_SIZE);
        cudaDeviceSynchronize();

        printf("Test 1 (Simple):\n");
        for (int i = 0; i < A_SIZE; ++i) printf("  out[%d] = %.1f\n", i, out[i]);
        cudaFree(a); cudaFree(b); cudaFree(out);
    }
    // ============ Test 2: Full ============
    {
        const int A_SIZE = 15, B_SIZE = 4;
        const int BPG = (A_SIZE + TPB - 1) / TPB;   // ceil(15/8) = 2
        float *a, *b, *out;
        cudaMallocManaged(&a,   A_SIZE * sizeof(float));
        cudaMallocManaged(&b,   B_SIZE * sizeof(float));
        cudaMallocManaged(&out, A_SIZE * sizeof(float));

        for (int i = 0; i < A_SIZE; ++i) a[i] = static_cast<float>(i);
        for (int j = 0; j < B_SIZE; ++j) b[j] = static_cast<float>(j);
        for (int i = 0; i < A_SIZE; ++i) out[i] = 0.0f;

        conv_kernel<<<BPG, TPB>>>(out, a, b, A_SIZE, B_SIZE);
        cudaDeviceSynchronize();

        printf("Test 2 (Full):\n");
        for (int i = 0; i < A_SIZE; ++i) printf("  out[%d] = %.1f\n", i, out[i]);
        cudaFree(a); cudaFree(b); cudaFree(out);
    }

    return 0;
}