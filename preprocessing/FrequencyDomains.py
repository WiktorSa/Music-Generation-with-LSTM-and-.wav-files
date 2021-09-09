import numpy as np
from scipy.fft import fft


def get_frequency_domains(data: np.ndarray, no_samples: int) -> np.ndarray:
    """
    Get frequency domains from data by using Fourier transforms

    :param data: data read from the .wav file
    :param no_samples: number of samples
    :return: numpy array containing the frequency domains of the song (real and imaginary part is separated)
    """

    samples = get_samples(data, no_samples)
    freq_domains = fft(samples)
    freq_domains_separated = np.empty(shape=no_samples * 2, dtype=np.float32)

    for i in range(0, no_samples * 2, 2):
        freq_domains_separated[i] = freq_domains[i // 2].real
        freq_domains_separated[i + 1] = freq_domains[i // 2].imag

    return freq_domains_separated


def get_samples(data: np.ndarray, no_samples: int) -> np.ndarray:
    """
    Get samples from data

    :param data: data read from the .wav file
    :param no_samples: number of samples
    :return: numpy array containing samples from data
    """

    sample_rate = data.shape[0] // no_samples
    samples = np.empty(shape=no_samples, dtype=np.float32)
    for i in range(no_samples):
        samples[i] = data[i * sample_rate]

    return samples
