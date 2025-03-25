#!/usr/bin/env bash
#PBS -N grayscale
#PBS -q classgpu
#PBS -l select=1:gpuname=ampere:ngpus=1:ncpus=1:mpiprocs=1:mem=1000mb
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -o grayscale.qsub_out

# change into submission directory
cd $PBS_O_WORKDIR

rm -rf build
mkdir -p build

# Configure the project with CMake
cmake -S . -B build \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_CUDA_ARCHITECTURES="80" \
  -DCMAKE_PREFIX_PATH="/apps/x86-64/apps/cuda_12.6.0" \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/build/install && \
  cmake --build build -j $(nproc) && \
  cmake --install build

rm -rf cpp/build
cmake -S cpp -B cpp/build -DCMAKE_PREFIX_PATH=$(pwd)/build/install/ && \
cmake --build cpp/build -j $(nproc)  && \
./cpp/build/my_project


