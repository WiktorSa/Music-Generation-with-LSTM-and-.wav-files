import numpy as np
from os import mkdir, listdir
from os.path import join, isdir
from preprocessing.PreprocessData import get_preprocessed_data


def generate_and_save_data(directory: str, sample_rate: int = 16000, sample_frequency: int = 4, len_window: int = 4000,
                  len_piece: int = 10, trial_size: float = 0.8, val_size: float = 0.1,
                  save_folder: str = 'data', seed: int = 1001) -> None:
    """
    Generate data from raw .wav files and save them in a given folder

    :param directory: directory with raw .wav files
    :param sample_rate: sample rate of all the songs in the given directory (in Hz)
    :param sample_frequency: how often to sample from a given song (in Hz)
    :param len_window: the length of the window of data
    :param len_piece: the length of one batch of data (in seconds)
    :param trial_size: the size of trial data
    :param val_size: the size of validation data
    :param save_folder: name of directory where data should be saved
    :param seed: seed used for RNG
    """

    files = [join(directory, f) for f in listdir(directory)]

    rng = np.random.default_rng(seed)
    rng.shuffle(files)

    train_indices = int(len(files) * trial_size)
    val_indices = int(len(files) * (trial_size + val_size))

    train_files = files[:train_indices]
    val_files = files[train_indices:val_indices]
    test_files = files[val_indices:]

    X_train, y_train = get_preprocessed_data(train_files, sample_rate, sample_frequency, len_window, len_piece)
    X_val, y_val = get_preprocessed_data(val_files, sample_rate, sample_frequency, len_window, len_piece)
    X_test, y_test = get_preprocessed_data(test_files, sample_rate, sample_frequency, len_window, len_piece)

    if not isdir(save_folder):
        mkdir(save_folder)

    with open(join(save_folder, "info_about_songs.txt"), 'w') as f:
        f.write("This songs are used for training:\n")
        for file in train_files:
            f.write(file + "\n")

        f.write("\nThis songs are used for validation:\n")
        for file in val_files:
            f.write(file + "\n")

        f.write("\nThis songs are used for test:\n")
        for file in test_files:
            f.write(file + "\n")

    np.savez(join(save_folder, 'data.npz'), X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
