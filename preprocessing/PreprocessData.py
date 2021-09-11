import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from preprocessing.GenerateFrequencyDomains import get_frequency_domains
from preprocessing.GeneratePiecesOfMusic import get_pieces_of_music
from preprocessing.GenerateSequences import get_sequence_X_and_y


def get_preprocessed_data(files: list, sample_rate: int = 16000, sample_frequency: int = 4, len_window: int = 4000,
                          len_piece: int = 10, seed: int = 1001) -> tuple[np.ndarray, np.ndarray]:
    """
    Get preprocessed data that can be used for training

    :param files: list of .wav files
    :param sample_rate: the sample rate of all songs (in Hz)
    :param sample_frequency: the frequency of sampling data used as a seed in our song
    :param len_window: the length of every window of data. For X the length of the window will be equal to
    2 * len_window. For y it will be equal to len_window
    :param len_piece: the length of one batch of data (in seconds)
    :param seed: seed for randomly shuffling batches
    :return: X and y in this shape: no_batches x no_windows x data
    """

    no_batches = get_no_batches(files, sample_rate, len_piece)
    no_windows = (len_piece * sample_rate) // len_window
    X = np.empty(shape=(no_batches, no_windows, len_window*2), dtype=np.float32)
    y = np.empty(shape=(no_batches, no_windows, len_window), dtype=np.float32)
    i = 0

    for file in files:
        _, data = wavfile.read(file)

        # Sample frequency domains from the whole song
        freq_domains = get_frequency_domains(data, sample_frequency, len_window)
        # Get the music divided into appriopriate parts
        pieces_of_music = get_pieces_of_music(data, sample_rate, len_piece)

        for piece_of_music in pieces_of_music:
            X_seq, y_seq = get_sequence_X_and_y(freq_domains, piece_of_music)
            freq_domains = get_frequency_domains(y_seq[-1])

            X[i] = X_seq
            y[i] = y_seq
            i += 1

    # Randomly shuffle the batches
    indices = np.arange(no_batches)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    X = X[indices]
    y = y[indices]

    return X, y


def get_no_batches(files: list, sample_rate: int, len_piece: int) -> int:
    """
    Get the number of batches in our data

    :param files: list of .wav files
    :param sample_rate: the sample rate of all songs (in Hz)
    :param len_piece: the length of one batch of data (in seconds)
    :return: the number of batches in our data
    """

    no_batches = 0
    for file in files:
        _, data = wavfile.read(file)
        no_batches += (data.shape[0] / sample_rate) // len_piece

    return int(no_batches)
