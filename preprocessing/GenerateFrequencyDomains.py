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

    for i in range(data.shape[0]):
        freq_domains_separated[i] = freq_domains[i].real
        freq_domains_separated[i + data.shape[0]] = freq_domains[i].imag

    return freq_domains_separated
