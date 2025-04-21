#include <convexa.h>

namespace ConvExa
{
template <typename T>
std::vector<std::complex<T>> host_dft(const std::vector<T> &signal)
{
    // Slight optimization, pre-reserve the size of the result
    size_t N = signal.size();
    std::vector<std::complex<T>> result;
    result.reserve(N);
    // For each FFT component
    for (int idx = 0; idx < static_cast<int>(N); idx++)
    {
        // For each signal element
        result.push_back({0.0, 0.0});
        for (int jdx = 0; jdx < static_cast<int>(N); jdx++)
        {
            // Generate complex number
            std::complex<T> exponential = std::exp(
                -(ConvExa::j * 2.0 * ConvExa::pi * static_cast<T>(idx) * static_cast<T>(jdx) 
                / static_cast<T>(N))    
            );
            // Append to the last element on the vector
            result.back() += static_cast<std::complex<T>>(signal[jdx]) * exponential;
        }
        // Summation is complete, move on to next FFT component
    }
    T scale = 1.0 / static_cast<T>(N);
    for (int i = 0; i < result.size(); ++i) {
        result[i] *= N;
    }
    return result;
}

template <>
std::vector<std::complex<float>> host_dft(const std::vector<float> &signal)
{
    size_t N = signal.size();
    std::vector<std::complex<float>> result;
    result.reserve(N);
    
    for (int idx = 0; idx < static_cast<int>(N); idx++)
    {
        result.push_back({0.0, 0.0});
        for (int jdx = 0; jdx < static_cast<int>(N); jdx++)
        {
            std::complex<float> exponential = std::exp(
                -(ConvExa::j_f * 2.0f * ConvExa::pi_f * static_cast<float>(idx) * static_cast<float>(jdx) 
                / static_cast<float>(N))    
            );
            result.back() += static_cast<std::complex<float>>(signal[jdx]) * exponential;
        }
    }

    float scale = 1.0 / static_cast<float>(N);
    for (int i = 0; i < result.size(); ++i) {
        result[i] *= N;
    }
    return result;
}
}
template <typename T>
float CXTiming::host_dft(const std::vector<T>& signal, std::vector<std::complex<T>>& output)
{
    auto start = std::chrono::high_resolution_clock::now();

    output = ConvExa::host_dft(signal);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (duration.count() > 0) {
        return duration.count() / 1000.0;
    } else {
        return 0;
    }
}
template float CXTiming::host_dft<double>(const std::vector<double>& signal, std::vector<std::complex<double>>& output);
template float CXTiming::host_dft<float>(const std::vector<float>& signal, std::vector<std::complex<float>>& output);