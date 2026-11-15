cmake_minimum_required(VERSION 3.18)  # 推荐使用 3.18 或更高版本

# 1. 定义项目，并启用 C++ 和 CUDA 语言
project(hello_cuda LANGUAGES CXX CUDA)

# 2. 设置 CUDA 的 C++ 标准（可选）
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 3. 关键步骤：指定目标 GPU 架构为 sm_120
set(CMAKE_CUDA_ARCHITECTURES 120)

# 4. 创建可执行文件，并加入你的 .cu 源文件
add_executable(output printID.cu)