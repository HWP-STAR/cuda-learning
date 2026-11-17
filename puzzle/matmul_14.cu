#include <cstdio>
#include <cuda_runtime.h>

constexpr int TPB = 3;   // 每 block 每维线程数，也是共享内存 tile 大小

__global__ void mm_kernel(float* out,const float* a,const float* b,int size)
{
    __shared__ float a_shared[TPB][TPB];
    __shared__ float b_shared[TPB][TPB];

    int i = blockIdx.x*blockDim.x + threadIdx.x; //row
    int j = blockIdx.y*blockDim.y + threadIdx.y; //col
    int local_i = threadIdx.x;
    int local_j= threadIdx.y;
    float acc = 0.0f;
    int num_tiles = (size + TPB -1)/TPB; // 向上取整,分块数

    for(int t = 0;t<num_tiles;++t)
    {
        int kk=t* TPB; //小分块 * 线程

        //行方向没有“分块滑动”，而列方向有
        // ===== ① 加载 a 的 [i, kk+local_j] → a_shared[local_i][local_j] =====
        if (i < size && kk + local_j < size) {
            a_shared[local_i][local_j] = a[i * size + (kk + local_j)];   // ★ 全局读
        } else {
            a_shared[local_i][local_j] = 0.0f;              // 越界填 0，不污染点积
        }
        // ===== ② 加载 b 的 [kk+local_i, j] → b_shared[local_i][local_j] =====
        if (kk + local_i < size && j < size) {
            b_shared[local_i][local_j] = b[(kk + local_i) * size + j];   // ★ 全局读
        } else {
            b_shared[local_i][local_j] = 0.0f;
        }

        // ★ 第一次屏障：所有线程都加载完，才能开始读共享内存
        __syncthreads();

                // ===== ③ 累加本轮 TPB 个乘积 =====
        for (int k = 0; k < TPB; ++k) 
        {
            acc += a_shared[local_i][k] * b_shared[k][local_j]; 
            //在 tile 循环外面初始化，得到的是整体结果，acc是叠加
        }

        // ★ 第二次屏障：所有线程都算完，下一轮加载才能覆盖共享内存
        __syncthreads();
    }

    //协会全局
    if(i < size && j < size) out[i* size + j] = acc;
}

// ---------- 辅助：构造测试数据并运行 ----------
void run_case(const char* name, int SIZE, dim3 grid, dim3 block) {
    float *a, *b, *out;
    cudaMallocManaged(&a,   SIZE * SIZE * sizeof(float));
    cudaMallocManaged(&b,   SIZE * SIZE * sizeof(float));
    cudaMallocManaged(&out, SIZE * SIZE * sizeof(float));

    // a[i][j] = i*SIZE + j；b = a^T，即 b[i][j] = a[j][i]
    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j) {
            a[i * SIZE + j] = static_cast<float>(i * SIZE + j);
            b[i * SIZE + j] = static_cast<float>(j * SIZE + i);
            out[i * SIZE + j] = 0.0f;
        }
    mm_kernel<<<grid, block>>>(out, a, b, SIZE);
    cudaDeviceSynchronize();

    printf("=== %s (SIZE=%d) ===\n", name, SIZE);
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j)
            printf("%7.0f ", out[i * SIZE + j]);
        printf("\n");
    }

    cudaFree(a); cudaFree(b); cudaFree(out);
}

int main() {
    // Test 1：SIZE=2，1×1 block，每 block 3×3 线程
    run_case("Matmul (Simple)", 2, dim3(1, 1), dim3(TPB, TPB));

    // Test 2：SIZE=8，ceil(8/3)=3 → 3×3 block，每 block 3×3 线程
    run_case("Matmul (Full)", 8, dim3(3, 3), dim3(TPB, TPB));

    return 0;
}