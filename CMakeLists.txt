# Minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(ConvExa CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Define source files (including subdirectories)
set(CONVEXA_SOURCES
    src/convexa.cpp
    src/convolutions/host_convolve.cpp
    # src/fft/fft.cpp
)

# Define the Python module
pybind11_add_module(
    convexa           # Module name
    bindings.cpp      # Bindings
    ${CONVEXA_SOURCES} # All source files
)

# Include directories
target_include_directories(convexa PRIVATE
    ${CMAKE_SOURCE_DIR}/include  # For convexa.h
    ${CMAKE_SOURCE_DIR}/src/convolutions  # For convolve.h
    ${CMAKE_SOURCE_DIR}/src/dfts          # For fft.h
)

# Optional: Optimize for performance
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(convexa PRIVATE -O3)
endif()

