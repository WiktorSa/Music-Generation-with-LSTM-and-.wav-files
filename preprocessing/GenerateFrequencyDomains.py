import sys

import numpy as np
from scipy.fft import fft


def get_frequency_domains(data: np.ndarray) -> np.ndarray:
    """
    Get frequency domains from data by using Fourier transforms

    :param data: data read from the .wav file
    :return: numpy array containing the frequency domains of the song (real and imaginary part is separated)
    """

    freq_domains = fft(data)
    freq_domains_separated = np.empty(shape=data.shape[0] * 2, dtype=np.float32)

    for i in range(0, data.shape[0] * 2, 2):
        try:
            freq_domains_separated[i] = freq_domains[i // 2].real
            freq_domains_separated[i + 1] = freq_domains[i // 2].imag
        except ValueError:
            print(freq_domains.shape)
            print(freq_domains_separated.shape)
            print(freq_domains[i // 2].real)
            sys.exit(-1)

    return freq_domains_separated
