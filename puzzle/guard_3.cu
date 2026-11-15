// guard.cu
#include <cstdio>
#include <cuda_runtime.h>

constexpr int N      = 4;   // 实际数据量
constexpr int THREADS = 8;  // 启动的线程数（故意多于数据量）

__global__ void map_guard(float* out, const float* a,int size)
{
    int idx = threadIdx.x;
    if(idx <size)
        out[idx] = a[idx] + 10.0f;
}

int main() {
    float *a, *out;
    cudaMallocManaged(&a,   N * sizeof(float));
    cudaMallocManaged(&out, N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        a[i]   = static_cast<float>(i);  // a = [0, 1, 2, 3]
        out[i] = 0.0f;
    }
    map_guard<<<1, THREADS>>>(out, a, N);  // 1 block × THREADS 线程
    cudaDeviceSynchronize();
    for (int i = 0; i < N; ++i)
        printf("out[%d] = %.1f\n", i, out[i]);

    cudaFree(a);
    cudaFree(out);
    return 0;
}