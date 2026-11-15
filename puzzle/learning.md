# cuda
<<<1, block>>>
第一个参数：grid 尺寸（有多少个 block）
第二个参数：block 尺寸（每个 block 有多少线程）

dim3 定义线程维度，默认是一维度

nvcc 并没有替代 g++/gcc，而是指挥它们。它负责处理 CUDA 特有的语法和 GPU 架构，而把标准的 C++ 代码编译工作“外包”给了你系统里现成的 GNU 编译器

每个SM中有多个block,一个 block 最多 1024 个线程
可以把它想象成一个只有 2048 个座位的报告厅（SM），虽然有32 个入口（Block 上限），但每个入口放进来的人（线程）总数加起来，绝对不能超过 2048 个座位。
**block和线程分离**
Block 是“软件概念”：它是一组线程的集合，是程序员在代码里划分的工作小组。一个 Block 里的线程可以共享同一个 SM 上的共享内存（Shared Memory），并能进行同步（__syncthreads）。

SM 是“硬件实体”：它才是真正装着 CUDA Core、Tensor Core、寄存器堆（Register File）、共享内存/L1 缓存的“芯片上的工厂”

