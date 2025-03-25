#include <convexa.h>

namespace ConvExa
{
std::vector<double> host_convolve(const std::vector<double> &signal, const std::vector<double> &kernel)
{
    size_t ol = signal.size() + kernel.size() - 1;
    std::vector<double> result;
    result.reserve(ol);

    for (size_t n = 0; n < ol; n++)
    {
        // for each output location
        double sum = 0;
        for (size_t i = 0; i < kernel.size(); i++)
        {
            if (n >= i && (n-i) < signal.size())
                sum += kernel[i] * signal[n - i];
        }
        result.push_back(sum);
    }
    return result;
}
}

float CXTiming::host_convolve(const std::vector<double>& signal, const std::vector<double>& kernel, std::vector<double>& output) {

    auto start = std::chrono::high_resolution_clock::now();
    output = ConvExa::host_convolve(signal, kernel);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (duration.count() > 0) {
        return duration.count() / 1000.0;
    } else {
        return 0;
    }
}
