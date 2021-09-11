import numpy as np
from scipy.fft import fft


def get_frequency_domains(data: np.ndarray, sample_frequency: int = 1, no_samples: int = None) -> np.ndarray:
    """
    Get frequency domains from data by using Fourier transforms

    :param data: data read from the .wav file
    :param sample_frequency: how often to sample data (in Hz).
    :param no_samples: number of samples to take from data. If None than no_samples = data.shape[0]
    :return: numpy array containing the frequency domains of the song (real and imaginary part is separated)
    """

    if no_samples is None:
        no_samples = data.shape[0]
        freq_domains = fft(data)
    else:
        samples = get_samples(data, sample_frequency, no_samples)
        freq_domains = fft(samples)

    freq_domains_separated = np.empty(shape=no_samples * 2, dtype=np.float32)

    for i in range(0, no_samples * 2, 2):
        freq_domains_separated[i] = freq_domains[i // 2].real
        freq_domains_separated[i + 1] = freq_domains[i // 2].imag

    return freq_domains_separated


def get_samples(data: np.ndarray, sample_frequency: int, no_samples: int) -> np.ndarray:
    """
    Get samples from data

    :param data: data read from the .wav file
    :param sample_frequency: how often to sample data (in Hz).
    :param no_samples: the number of samples to take from data
    :return: numpy array containing samples from data
    """

    samples = np.empty(shape=no_samples, dtype=np.float32)
    for i in range(no_samples):
        samples[i] = data[i * sample_frequency]

    return samples
