import numpy as np


if __name__ == "__main__":
    signal = [2, 1, 3, 5, 7, 6, 4]
    result = np.fft.fft(signal)
    print(result)