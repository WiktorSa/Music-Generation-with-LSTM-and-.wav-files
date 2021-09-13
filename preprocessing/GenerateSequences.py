import numpy as np
from preprocessing.GenerateFrequencyDomains import get_frequency_domains


def generate_sequences(data: np.ndarray, len_window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate sequences of data that can be used for training

    :param data: data read from .wav file
    :param len_window: the length of one window of data
    :return: numpy array containing the data divided into sequences
    """

    # Because we cannot predict the first part of the song the no_sequences will be smaller by 1
    no_sequences = int(data.shape[0] / len_window) - 1
    x_sequences = np.empty(shape=(no_sequences, len_window * 2), dtype=np.float32)
    y_sequences = np.empty(shape=(no_sequences, len_window))

    for i in range(no_sequences):
        x_sequences[i] = get_frequency_domains(data[i * len_window:(i + 1) * len_window])
        y_sequences[i] = data[(i + 1) * len_window:(i + 2) * len_window]

    return x_sequences, y_sequences
