import sys

import numpy as np
from scipy.fft import fft


def get_frequency_domains(data: np.ndarray) -> np.ndarray:
    """
    Get frequency domains from data by using Fourier transforms

    :param data: data read from the .wav file
    :return: numpy array containing the frequency domains of the song (real and imaginary parts are separated)
    """

    freq_domains = fft(data, norm='forward')
    freq_domains_separated = np.empty(shape=data.shape[0] * 2, dtype=np.float64)
    freq_domains_separated[:data.shape[0]] = freq_domains.real
    freq_domains_separated[data.shape[0]:] = freq_domains.imag

    return freq_domains_separated
