import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft
from os import listdir
from os.path import join


def get_preprocessed_data(directory: str, sample_rate: int = 16000, sample_frequency: float = 0.25) -> np.ndarray:
    """
    Get preprocessed data that can be used for training

    :param directory: the location of the folder with raw audio
    :param sample_rate: the sample rate of all songs
    :param sample_frequency: the frequency of sampling used
    :return: numpy array containing frequency domains and something more (finish later)
    """

    files = [join(directory, f) for f in listdir(directory)]
    no_samples = int(sample_rate * sample_frequency)

    #  Frequency domains
    freq_domains = np.empty(shape=len(files), dtype=np.object)
    for i in range(len(files)):
        _, data = wavfile.read(files[i])
        samples = get_samples(data, no_samples)
        freq_domains[i] = get_frequency_domains(samples, no_samples)  # Add torch.Tensor ???


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


def get_frequency_domains(samples: np.ndarray, no_samples: int) -> np.ndarray:
    """
    Get frequency domains from data by using Fourier transforms

    :param samples: samples got from the function get_samples
    :param no_samples: number of samples
    :return: numpy array containing the frequency domains of the song (real and imaginary part is separated)
    """

    freq_domains = fft(samples)
    freq_domains_separated = np.empty(shape=no_samples * 2, dtype=np.float32)

    for i in range(0, no_samples * 2, 2):
        freq_domains_separated[i] = freq_domains[i // 2].real
        freq_domains_separated[i + 1] = freq_domains[i // 2].imag

    return freq_domains_separated


if __name__ == '__main__':
    get_preprocessed_data('../raw_audio')
