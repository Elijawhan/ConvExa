#!/usr/bin/env bash
#PBS -N convexa
#PBS -q gpu
#PBS -l select=1:gpuname=ampere:ngpus=1:ncpus=1:mpiprocs=1:mem=1000mb
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -o convexa.qsub_out

# change into submission directory
cd $PBS_O_WORKDIR

CUDA_HOME=/apps/x86-64/apps/cuda_12.6.0

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
rm -f *.ncu-rep
cmake -S cpp -B cpp/build -DCMAKE_PREFIX_PATH=$(pwd)/build/install/ && \
cmake --build cpp/build -j $(nproc)  && \
module load cuda && \
ncu -o my_project ./cpp/build/my_project && \
${CUDA_HOME}/bin/nsys profile -o my_project.nsys-rep --force-overwrite true \
  ./cpp/build/my_project


