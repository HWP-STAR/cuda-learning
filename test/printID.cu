#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void printID()
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from block %d, thread %d-> global ID %d\n", blockIdx.x, threadIdx.x,id);
}

int main()
{
    printID<<<3,4>>>();
    cudaDeviceSynchronize();
    return 0;
}