import numpy as np


def get_pieces_of_music(data: np.ndarray, sample_rate: int, len_piece: int) -> np.ndarray:
    """
    Get pieces of music from a single song. The length of the piece is determined by the length_of_pieces argument

    :param data: data read from .wav file
    :param sample_rate: the sample rate of the song
    :param len_piece: the length of the single piece
    :return: numpy array containing pieces of music
    """

    no_pieces = int((data.shape[0] / sample_rate) // len_piece)
    len_data = sample_rate * len_piece

    pieces_of_music = np.empty(shape=(no_pieces, len_data), dtype=np.float32)

    for i in range(no_pieces):
        pieces_of_music[i, :] = data[i*len_data:(i+1)*len_data]

    return pieces_of_music
