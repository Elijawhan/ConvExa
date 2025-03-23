#include <convexa.h>

namespace ConvExa
{
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
}

namespace CETiming
{
double host_dft(const std::vector<double>& signal, std::vector<std::complex<double>>& output)
{
    auto start = std::chrono::high_resolution_clock::now();

    output = ConvExa::host_dft(signal);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (duration.count() > 0) {
        return duration.count() * 1000.0;
    } else {
        return 0;
    }
}
} //end namespace 