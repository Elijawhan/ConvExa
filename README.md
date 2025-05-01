# ConvExa: Convolution for Exascale Solutions
ConvExa is a library which uses the CUDA 12.6 and cuFFT libraries to implement Optimized methods for Convolution.
Each of these implementations of Convolution have different execution times based on the size of the Kernels and Signals. 
This feature is available with the Batch Convolve function shown here.
```cpp
std::vector<std::vector<T>> ConvExa::batch_convolve(std::vector<std::vector<T>>, std::vector<std::vector<T>>);
```
The Batch Convolve function uses a system referred to as the Smart batch Dispatcher to choose an implementation which will execute the fastest based on the input shapes.


## Dependencies
- CUDA 12.6+
- cuFFT 12+

## Build Instructions (ASAX)
Building Convexa on ASAX is as easy as running the run_me.sh file available in the root directory of the project. 
This will build the library and the accompanying CPP test project, as well as run the test project.

To Add Convexa to your own project after building the library, you should:
1. Add `find_package(convexa REQUIRED)` and `target_link_libraries(my_project PRIVATE convexa::convexa)` to your existing CMakeLists.txt.
2. Build as you normally would.

## Finding Data to use.
Convexa comes with an accompanying `HELP` namespace, which contains many functions which can assist users in trying out the library. Namely, the Generate Random Vector function, which can create a randomly generated vector of a chosen size which can be used as a sample input.
```cpp
std::vector<T> generate_random_vector(size_t size, T min, T max);
```
Additionally, To use the library in audio processing, Helper functions for reading and writing .WAV Files are included. This means you can also use your own WAV Files to test on. Currently, the Help functions only support monochannel WAV Files. 
Example WAV Files are included for use in Testing in the sample project.
```cpp
std::vector<int16_t> data;
HELP::wav_hdr hdr = HELP::read_wav("./cpp/audio/my_audio.wav", &data);
// Convolutions here
HELP::write_wav("./cpp/audio/my_audio_convolved.wav", HELP::vec_cast<T, int16_t>(myResult), hdr);
```
[Audacity](https://www.audacityteam.org), which is Free and Open Source Software is recommended for the recording of your own audio.
## Key Parameters and Configuration Options
- There are no Parameters or additional configuration options needed in this iteration of ConvExa
## Repository Structure Overview
All .h Files pertaining to the library are stored in the `include` directory. Implementation code is stored in `src`, but is broken up into `convolutions` and `dfts` directories which contain implementations for Convolution methods and dft methods respectively.
`scripts` directory contains build scripts for different systems. Currently `configure_build_run.sh` is the only fully tested one.
`python` directory is for future iterations of this project which will create python bindings for ConvExa.
`cpp` directory is an example project, and the test project for ConvExa, and contains procedures for testbenching and validation of ConvExa, as well as example code usage and CMakeLists.txt for consuming projects.