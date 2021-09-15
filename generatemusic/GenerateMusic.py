from os import mkdir, listdir
from os.path import join, isdir
import numpy as np
from joblib import load
from scipy.io import wavfile
import torch
from preprocessing.GenerateFrequencyDomains import get_frequency_domains
from model import MusicModel


def generate_music(seed_dir: str = 'seeds', model_dir: str = 'model_weights', normalizer_dir: str = 'normalizer',
                   save_dir: str = 'generated_songs', start_second: int = 5, len_piece: int = 10,
                   sample_rate: int = 16000, input_size: int = 8000, hidden_size: int = 2048, output_size: int = 4000,
                   dropout: float = 0.2) -> None:
    """
    Generate music based on a sample from songs

    :param seed_dir: Directory with audio from which we will take samples as a seed for the generator
    :param model_dir: directory with saved weights of the model
    :param normalizer_dir: directory with saved normalizer
    :param save_dir: directory where generated music should be saved
    :param start_second: from which second of song the function should take a sample
    :param len_piece: the length of generated song (in seconds)
    :param sample_rate: sample rate of songs that were used in training
    :param input_size: input size that was used in training
    :param hidden_size: hidden size that was used in training
    :param output_size: output size that was used in training
    :param dropout: value of dropout that was used in training (linear layer)
    """

    # There is no need to use gpu here
    # But if you want to use gpu change the value to 'cuda'
    device = 'cpu'

    files = [join(seed_dir, f) for f in listdir(seed_dir)]
    normalizer = load(join(normalizer_dir, 'normalizer.joblib'))
    model = MusicModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout=dropout,
                       batch_size=1).to(device)
    model.load_state_dict(torch.load(join(model_dir, 'model_weights.pth'), map_location=torch.device(device)))
    model.eval()

    if not isdir(save_dir):
        mkdir(save_dir)

    for i, file in enumerate(files, 1):
        model.init_hidden(device)

        sample_rate_song, data = wavfile.read(file)

        # data is only used as a seed. So the function is written in such a way that we can take a seed from any song
        # (although it's advisable to use songs with proper sample rate and number of channels)
        data = data.reshape(-1)
        seed = data[sample_rate_song * start_second:sample_rate_song * start_second + output_size]

        freq_domains = get_frequency_domains(seed).reshape(1, -1)
        freq_domains = normalizer.transform(freq_domains)
        song = np.empty(shape=(int(len_piece * sample_rate / output_size), output_size), dtype=np.int16)
        for j in range(song.shape[0]):
            with torch.no_grad():
                x = torch.from_numpy(freq_domains).to(device)
                # Conversion needs to be made because we save in int16 format
                song[j] = model(x).numpy().astype(np.int16)

            freq_domains = get_frequency_domains(song[j]).reshape(1, -1)
            freq_domains = normalizer.transform(freq_domains)

        # Save song
        song = song.reshape(-1)
        wavfile.write(join(save_dir, 'generated_song_' + str(i) + '.wav'), sample_rate, song)
