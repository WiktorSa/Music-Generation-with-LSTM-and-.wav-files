import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from os import listdir
from os.path import join
from FrequencyDomains import get_frequency_domains
from PiecesOfMusic import get_pieces_of_music


def get_preprocessed_data(files: list, sample_rate: int = 16000, sample_frequency: int = 4,
                          length_of_pieces: int = 10) -> tuple[list, list]:
    """
    Get preprocessed data that can be used for training

    :param files: list of .wav files
    :param sample_rate: the sample rate of all songs (in Hz)
    :param sample_frequency: the frequency of sampling data used in getting frequency domains (in Hz)
    :param length_of_pieces: the length of pieces that the songs will be divided into (in seconds).
    :return: X and y - both in a form of a list
    """

    no_samples = sample_rate // sample_frequency

    for i in range(1):
        _, data = wavfile.read(files[i])

        freq_domains = get_frequency_domains(data, no_samples)
        pieces_of_music = get_pieces_of_music(data, sample_rate, length_of_pieces)
        print(freq_domains.shape)

    return [], []


if __name__ == '__main__':
    filesa = [join('../raw_audio', f) for f in listdir('../raw_audio')]
    get_preprocessed_data(filesa)
