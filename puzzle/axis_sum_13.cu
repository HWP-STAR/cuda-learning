#include <cstdio>
#include <cuda_runtime.h>

constexpr int TPB = 8;   // 每 block 线程数，也是每段处理的列数

__global__ void axis_sum_kernel(float* out,const float* a,int size)
{
    __shared__ float cache[TPB];
    int batch = blockIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x; //全局col
    int local_i = threadIdx.x;

    // global to shared,other is 0
    if(col < size){
        cache[local_i] = a[batch*size+col];
    }else{
        cache[local_i] = 0.0f;
    }
    __syncthreads();

    //归约
    for (int stride=TPB/2;stride >0; stride >>= 1)
    {
        if(local_i< stride) cache[local_i] += cache[local_i+stride];
        __syncthreads();
    }

    //写出
    if(local_i == 0)
    {
        int num_segments = (size + TPB -1)/TPB; //上取整，一个size要分成都是block
        out[batch*num_segments + blockIdx.x] = cache[0];
    }
}

int main() {
    const int BATCH = 4;
    const int SIZE  = 6;
    const int NUM_SEGMENTS = (SIZE + TPB - 1) / TPB;   // = 1

    float *a, *out;
    cudaMallocManaged(&a,   BATCH * SIZE * sizeof(float));
    cudaMallocManaged(&out, BATCH * NUM_SEGMENTS * sizeof(float));

    // 初始化 a = arange(24).reshape(4, 6)，行优先
    for (int i = 0; i < BATCH * SIZE; ++i)
        a[i] = static_cast<float>(i);
    for (int i = 0; i < BATCH * NUM_SEGMENTS; ++i)
        out[i] = 0.0f;

    axis_sum_kernel<<<dim3(NUM_SEGMENTS,BATCH),TPB>>>(out,a,SIZE);
    cudaDeviceSynchronize();

    // 打印
    for (int b = 0; b < BATCH; ++b) {
        for (int s = 0; s < NUM_SEGMENTS; ++s) {
            printf("out[%d][%d] = %.1f\n", b, s, out[b * NUM_SEGMENTS + s]);
        }
    }
    cudaFree(a);
    cudaFree(out);
    return 0;
}
