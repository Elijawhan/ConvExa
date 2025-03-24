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
constexpr cuda::std::complex<double> j_d(0.0, 1.0);
constexpr cuda::std::complex<float> j_f(0.0, 1.0);
constexpr double pi = 3.141592653589793238462643383279;
constexpr float pi_f = 3.141592653589793238462643383279f;


std::vector<double> host_convolve(const std::vector<double>& signal, const std::vector<double>& kernel);
std::vector<std::complex<double>> host_dft(const std::vector<double>& signal);
} //End namespace ConvExa

namespace CETiming
{
double host_convolve(const std::vector<double>& signal, const std::vector<double>& kernel, std::vector<double>& output);
double host_dft(const std::vector<double>& signal, std::vector<std::complex<double>>& output);
}

