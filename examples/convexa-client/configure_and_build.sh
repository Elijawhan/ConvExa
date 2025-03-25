#!/usr/bin/env bash

rm -rf build

export LD_LIBRARY_PATH=

cmake -DCMAKE_INSTALL_PREFIX=`pwd`/build/install \
  -DCMAKE_PREFIX_PATH="~/ConvExa/build/install;/apps/x86-64/apps/cuda_12.6.0" \
  -DCMAKE_VERBOSE_MAKEFILE=ON \
  -S . -B build && \
  cmake --build build
