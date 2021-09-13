import numpy as np
from os import mkdir, listdir
from os.path import join, isdir
from sklearn.preprocessing import Normalizer
from joblib import dump
from preprocessing.PreprocessData import get_preprocessed_data


def generate_and_save_data(directory: str, sample_rate: int = 16000, len_song: int = 120, len_sample: float = 0.25,
                trial_size: float = 0.6, val_size: float = 0.2, save_folder: str = 'data',
                           save_norm: str = 'normalizer', seed: int = 1001) -> None:
    """
    Generate data from raw .wav files and save them in a given folder

    :param directory: directory with raw .wav files
    :param sample_rate: sample rate of all the songs in the given directory (in Hz)
    :param len_song: the length of the song (in seconds). If the song is longer than take a sample of given length
    :param len_sample: the length of one sample of data (in seconds)
    :param trial_size: the size of trial data
    :param val_size: the size of validation data
    :param save_folder: name of directory where data should be saved
    :param save_norm: name of directory where normalizer should be saved
    :param seed: seed
    """

    files = [join(directory, f) for f in listdir(directory)]

    rng = np.random.default_rng(seed)
    rng.shuffle(files)

    train_indices = int(len(files) * trial_size)
    val_indices = int(len(files) * (trial_size + val_size))

    train_files = files[:train_indices]
    val_files = files[train_indices:val_indices]
    test_files = files[val_indices:]

    X_train, y_train = get_preprocessed_data(train_files, sample_rate, len_song, len_sample)
    X_val, y_val = get_preprocessed_data(val_files, sample_rate, len_song, len_sample)
    X_test, y_test = get_preprocessed_data(test_files, sample_rate, len_song, len_sample)

    # Normalize data
    transformer = Normalizer()
    X_train = transformer.fit_transform(X_train)
    X_val = transformer.transform(X_train)
    X_test = transformer.transform(X_test)

    # Save data
    if not isdir(save_folder):
        mkdir(save_folder)

    if not isdir(save_norm):
        mkdir(save_norm)

    np.savez_compressed(join(save_folder, 'data.npz'), X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

    dump(transformer, join(save_norm, 'normalizer.joblib'))
