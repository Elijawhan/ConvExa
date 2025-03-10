#include "../convexa.h"

std::vector<std::complex<double>> host_dft(const std::vector<double> &signal)
{
    size_t N = signal.size();
    std::vector<std::complex<double>> result;
    result.reserve(N);
    
    for (int idx = 0; idx < static_cast<int>(N); idx++)
    {
        result.push_back({0.0, 0.0});
        for (int jdx = 0; jdx < static_cast<int>(N); jdx++)
        {
            std::complex<double> exponential = std::exp(
                -(ConvExa::j * 2.0 * ConvExa::pi * static_cast<double>(idx) * static_cast<double>(jdx) 
                / static_cast<double>(N))    
            );
            result.back() += static_cast<std::complex<double>>(signal[jdx]) * exponential;
        }
    }
    return result;
}