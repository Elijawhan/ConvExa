cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the installed convexa library
find_package(convexa REQUIRED)
set(CMAKE_PREFIX_PATH "/apps/x86-64/apps/cuda_12.6.0")
set(CMAKE_CUDA_ARCHITECTURES "80")
set(CMAKE_CUDA_COMPILER "/apps/x86-64/apps/cuda_12.6.0/bin/nvcc")
find_package(CUDAToolkit REQUIRED)

# Add your executable
add_executable(my_project main.cpp)
target_link_libraries(my_project PRIVATE CUDA::cudart)
# Link convexa to your executable
target_link_libraries(my_project PRIVATE convexa::convexa)
