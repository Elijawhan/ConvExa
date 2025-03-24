cmake_minimum_required(VERSION 3.15)
project(convexa VERSION 1.0.0
        LANGUAGES CXX CUDA) # Add a version
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_PREFIX_PATH "/apps/x86-64/apps/cuda_12.6.0")
set(CMAKE_CUDA_ARCHITECTURES "80")
set(CMAKE_CUDA_COMPILER "/apps/x86-64/apps/cuda_12.6.0/bin/nvcc")



find_package(CUDAToolkit REQUIRED)
# Create the library
add_library(convexa
  src/convexa.cpp
  src/convolutions/host_convolve.cpp
  src/dfts/host_dft.cpp
)

target_link_libraries(convexa PRIVATE CUDA::cudart)

# Install the library
install(TARGETS convexa
        EXPORT convexaTargets
        LIBRARY DESTINATION lib)

# Generate and install the config file
install(EXPORT convexaTargets
        FILE convexa-config.cmake
        NAMESPACE convexa::
        DESTINATION lib/cmake/convexa)

# Install the headers
install(FILES include/convexa.h include/cxkernels.h
        DESTINATION include)


# Generate libconvexaTargets.cmake for export
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/convexa-config-version.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY AnyNewerVersion
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/convexa-config-version.cmake"
        DESTINATION lib/cmake/convexa)

# Set the include directories property
target_include_directories(convexa PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>) # important for install

# Set the install prefix.
set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE PATH "Install prefix" FORCE)

# Set the library output name.
set_target_properties(convexa PROPERTIES OUTPUT_NAME "convexa")
