#include <cstdio>
#include <cuda_runtime.h>

constexpr int SIZE = 5;   // 数据是 SIZE × SIZE
constexpr int TPB  = 3;   // 每个维度每 block 3 个线程
constexpr int BPG  = 2;   // 每个维度 2 个 block

__global__ void map_block2d(float* out,const float* a, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;    // 行方向全局编号
    int j = blockIdx.y * blockDim.y + threadIdx.y;    // 列方向全局编号

    if(i < size && j < size)
    {
        int idx = i*size + j;
        out[idx] = a[idx] + 10.f;
    }

}

int main() {
    float *a, *out;
    cudaMallocManaged(&a,   SIZE * SIZE * sizeof(float));
    cudaMallocManaged(&out, SIZE * SIZE * sizeof(float));

    // a = 5×5 全 1，out 清零
    for (int k = 0; k < SIZE * SIZE; ++k) {
        a[k]   = 1.0f;
        out[k] = 0.0f;
    }

    map_block2d<<<dim3(BPG,BPG),dim3(TPB,TPB)>>>(out,a,SIZE);
    cudaDeviceSynchronize();

    // 按 2D 形式打印
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j)
            printf("%5.1f ", out[i * SIZE + j]);
        printf("\n");
    }

    cudaFree(a);
    cudaFree(out);
    return 0;
}