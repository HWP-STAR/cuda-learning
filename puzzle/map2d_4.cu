#include <cstdio>
#include <cuda_runtime.h>

constexpr int SIZE    = 2;   // 数据是 SIZE × SIZE 的方阵
constexpr int THREADS = 3;   // 每个维度启动 3 个线程，共 3×3 = 9 个

__global__ void map2d(float* out, const float* a, int size)
{
    int i = threadIdx.x;   // 行
    int j = threadIdx.y;   // 列

    if(i< size && j < size)
    {
        int idx = i * size + j;  // 计算一维索引
        out[idx] = a[idx] + 10.0f;
    }
}

int main()
{
    float *a, *out;
    cudaMallocManaged(&a,   SIZE * SIZE * sizeof(float));
    cudaMallocManaged(&out, SIZE * SIZE * sizeof(float));

    for (int i = 0; i < SIZE * SIZE; ++i) {
        a[i]   = static_cast<float>(i);
        out[i] = 0.0f;
    }
    dim3 threads(THREADS, THREADS);  // 2D 线程块
    map2d<<<1, threads>>>(out, a, SIZE);  // 1 block × THREADS×THREADS 线程
    cudaDeviceSynchronize();

    // 按 2D 形式打印
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            printf("out[%d][%d] = %.1f\n", i, j, out[i * SIZE + j]);
        }
    }

    cudaFree(a);
    cudaFree(out);
    return 0;

}