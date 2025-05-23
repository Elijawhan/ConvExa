cmake_minimum_required(VERSION 3.18)
project(convexa VERSION 1.0.0
        LANGUAGES CXX CUDA) # Add a version
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDAToolkit REQUIRED)
# Create the library
add_library(convexa
  src/convexa.cpp
  src/convolutions/host_convolve.cpp
  src/convolutions/device_convolve.cu
  src/convolutions/device_convolve_overlap_save.cu
  src/convolutions/device_convolve_fft.cu
  src/convolutions/batch_convolve.cu
  src/dfts/host_dft.cpp
  src/dfts/device_dft.cu
  src/dfts/device_fft_radix2.cu
  src/dfts/host_fft_radix2.cpp
  src/dfts/cufft.cu
)

target_link_libraries(convexa PRIVATE CUDA::cudart CUDA::cufft)

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
install(FILES include/convexa.h include/cxkernels.h include/helper.h
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
