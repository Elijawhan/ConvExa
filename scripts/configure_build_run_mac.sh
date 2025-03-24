#!/bin/bash

# Script to configure, build, and install the convexa library using modern CMake syntax.

# Create build directory
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

# install the python module into the virtual environment.
# Create virtual environment in the install directory for the python module.
python3 -m venv --system-site-packages build/install/convexa

# install the python module into the virtual environment.
source build/install/convexa/bin/activate
export CMAKE_INSTALL_PREFIX=`pwd`/build/install
export CMAKE_PREFIX_PATH=`pwd`/build/install
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/usr/local/lib/python3.10/dist-packages/pybind11 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/build/install/lib:/apps/x86-64/apps/cuda_12.6.0/lib64
python -m pip install ./python
python -m pip install numpy

# run test
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/build/install/lib
python python/examples/precision_valid.py

