import numpy as np


def get_pieces_of_music(data: np.ndarray, sample_rate: int, length_of_pieces: int) -> np.ndarray:
    """
    Get pieces of music from a single song. The length of the piece is determined by the length_of_pieces argument

    :param data: data read from .wav file
    :param sample_rate: the sample rate of the song
    :param length_of_pieces: the length of the piece
    :return: numpy array containing pieces of music
    """
    no_pieces = int((data.shape[0] / sample_rate) // length_of_pieces)
    length_of_data = sample_rate * length_of_pieces

    pieces_of_music = np.empty(shape=[no_pieces, length_of_data], dtype=np.object)

    for i in range(no_pieces):
        pieces_of_music[i, :] = data[i*length_of_data:(i+1)*length_of_data]

    return pieces_of_music
