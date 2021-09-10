import numpy as np
from FrequencyDomains import get_frequency_domains


def get_sequence_X_and_y(first_piece: np.ndarray, pieces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Get X and y sequences from one piece of music

    :param first_piece: piece that will be used to predict first y sequence
    :param pieces: pieces of music
    :return:
    """
    length_of_sequence_X = int(first_piece.shape[0])
    length_of_sequence_y = first_piece.shape[0] // 2
    no_sequences = pieces.shape[0] // length_of_sequence_y

    X_seq = np.empty(shape=(no_sequences, length_of_sequence_X), dtype=np.float32)
    y_seq = np.empty(shape=(no_sequences, length_of_sequence_y), dtype=np.float32)

    # y_seq - just divide pieces into proper parts:
    for i in range(no_sequences):
        y_seq[i] = pieces[i * length_of_sequence_y:(i + 1) * length_of_sequence_y]

    # X_seq - X_seq[0] is equal to first_piece. The rest of X_seq is equal to appriopriate fourier transforms
    # of pieces
    X_seq[0] = first_piece
    for i in range(1, no_sequences):
        X_seq[i] = get_frequency_domains(y_seq[i - 1])

    return X_seq, y_seq
