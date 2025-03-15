# Minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(convexa CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Define source files (including subdirectories)
set(CONVEXA_SOURCES
    convexa.cpp
    convolutions/host_convolve.cpp
    dfts/host_dft.cpp
)

# Define the Python module
pybind11_add_module(
    convexa           # Module name
    bindings.cpp      # Bindings
    ${CONVEXA_SOURCES} # All source files
)
target_link_libraries(convexa PRIVATE Python3::Python)

# Include directories
target_include_directories(convexa PRIVATE
    ${CMAKE_SOURCE_DIR}  # For convexa.h
    ${CMAKE_SOURCE_DIR}/convolutions  # For convolve.h
    ${CMAKE_SOURCE_DIR}/dfts          # For fft.h
)

# Optional: Optimize for performance
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(convexa PRIVATE -O3)
endif()

install(TARGETS convexa
  DESTINATION ${PYTHON_SITE_PACKAGES}/convexa) # Important install line.