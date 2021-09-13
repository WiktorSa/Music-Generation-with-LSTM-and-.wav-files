import numpy as np
from scipy.io import wavfile
from preprocessing.GenerateSequences import generate_sequences


def get_preprocessed_data(files: list, sample_rate: int = 16000, len_song: int = 120, len_sample: float = 0.25) \
        -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess data which includes:
    1. Sampling from the whole song
    2. Getting frequency domains for every sample
    3. Building appriopriate x and y sequences

    :param files: list of .wav files
    :param sample_rate: the sample rate of all songs (in Hz)
    :param len_song: the length of the song (in seconds). If the song is longer than take a sample of given length
    :param len_sample: the length of one sample of data (in seconds)
    :return: x and y sequences
    """

    no_samples_per_song = int(len_song / len_sample)
    no_samples_overall = no_samples_per_song * len(files)
    len_window = int(sample_rate * len_sample)
    x = np.empty(shape=(no_samples_overall, len_window * 2), dtype=np.float32)
    y = np.empty(shape=(no_samples_overall, len_window), dtype=np.float32)

    for i, file in enumerate(files):
        _, data = wavfile.read(file)

        # Sample from data (because first second of the song doesn't contain a lof of info we skip it)
        data = data[sample_rate:sample_rate * len_song + len_window + sample_rate]

        # Generate sequences
        x_seq, y_seq = generate_sequences(data, len_window)

        x[i * no_samples_per_song:(i + 1) * no_samples_per_song] = x_seq
        y[i * no_samples_per_song:(i + 1) * no_samples_per_song] = y_seq

    return x, y
