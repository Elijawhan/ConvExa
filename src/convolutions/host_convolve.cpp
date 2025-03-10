#include "../convexa.h"
std::vector<double> host_convolve(const std::vector<double> &signal, const std::vector<double> &kernel)
{
    size_t ol = signal.size() + kernel.size() - 1;
    std::vector<double> result;
    result.reserve(ol);

    for (size_t n = 0; n < ol; n++)
    {
        // for each output location
        double sum = 0;
        for (size_t i = kernel.size(); i-- > 0;)
        {
            if (n >= i)
                sum += kernel[i] * signal[n - i];
        }
        result.push_back(sum);
    }

    return result;
}
