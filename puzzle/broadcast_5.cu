#include <cstdio>
#include <cuda_runtime.h>

constexpr int SIZE    = 2;   // 输出是 SIZE × SIZE
constexpr int THREADS = 3;   // 每个维度 3 个线程，共 9 个

__global__ void broadcast_add(float* out, const float* a,const float* b,int size)
{
    int i = threadIdx.x;
    int j = threadIdx.y;

    if(i < size && j < size)
    {
        out[i * size + j] = a[i] + b[j];
    }
}

int main()
{
    float *a, *b, *out;
    cudaMallocManaged(&a,   SIZE * sizeof(float));        // (SIZE, 1) → SIZE 个元素
    cudaMallocManaged(&b,   SIZE * sizeof(float));        // (1, SIZE) → SIZE 个元素
    cudaMallocManaged(&out, SIZE * SIZE * sizeof(float)); // (SIZE, SIZE)

    for(int i=0;i<SIZE;++i) a[i] = static_cast<float>(i);
    for(int i=0;i<SIZE;++i) b[i] = static_cast<float>(i);

    for(int k=0;k<SIZE;++k) out[k] = 0.0f;

    broadcast_add<<<1,dim3(THREADS, THREADS)>>>(out, a, b, SIZE);
    cudaDeviceSynchronize();

    // 按 2D 形式打印
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            printf("out[%d][%d] = %.1f\n", i, j, out[i * SIZE + j]);
        }
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(out);
    return 0;
}