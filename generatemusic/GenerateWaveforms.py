import numpy as np
from scipy.fft import ifft


def generate_waveforms(data: np.ndarray) -> np.ndarray:
    """
    Generate waveforms from frequency domains

    :param data: frequency domains (where first n/2 examples consist of real values and the rest consists
    of imaginary values)
    :return: numpy array containing the waveforms
    """

    waveform_length = int(data.shape[0] // 2)

    real_values = data[:waveform_length]
    imag_values = data[waveform_length:]

    freq_domains = np.empty(shape=waveform_length, dtype=np.complex128)
    freq_domains.real = real_values
    freq_domains.imag = imag_values

    waveform = ifft(freq_domains)

    return waveform
