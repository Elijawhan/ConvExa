#pragma once

#include <vector>
#include <complex>
#include <cmath>
#include <math.h>
#include <chrono>

void hello_amazing_convexa_world();


namespace ConvExa
{
    constexpr std::complex<double> j(0.0, 1.0);
    constexpr double pi = 3.141592653589793238462643383279;
    constexpr float pi_f = 3.141592653589793238462643383279f;

    template <typename T>
    std::vector<T> host_convolve(const std::vector<T>& signal, const std::vector<T>& kernel);
    std::vector<std::complex<double>> host_dft(const std::vector<double>& signal);

} //End namespace ConvExa

namespace CXTiming
{
    template <typename T>
    float host_convolve(const std::vector<T>& signal, const std::vector<T>& kernel, std::vector<T>& output);
    float host_dft(const std::vector<double>& signal, std::vector<std::complex<double>>& output);
}

