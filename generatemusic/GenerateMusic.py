from os import mkdir, listdir
from os.path import join, isdir
import numpy as np
from joblib import load
from scipy.io import wavfile
import torch
from preprocessing.GenerateFrequencyDomains import get_frequency_domains
from model import MusicModel
from generatemusic.GenerateWaveforms import generate_waveforms


def generate_music(seed_dir: str = 'seeds', model_dir: str = 'model_weights', save_dir: str = 'generated_songs',
                   start_second: int = 5, len_piece: int = 10, sample_rate: int = 16000, input_size: int = 8000,
                   hidden_size: int = 2048, output_size: int = 8000, dropout: float = 0.2) -> None:
    """
    Generate music based on a sample from songs

    :param seed_dir: Directory with audio from which we will take samples as a seed for the generator
    :param model_dir: directory with saved weights of the model
    :param save_dir: directory where generated music should be saved
    :param start_second: from which second of song the function should take a sample
    :param len_piece: the length of generated song (in seconds)
    :param sample_rate: sample rate of songs that were used in training
    :param input_size: input size that was used in training
    :param hidden_size: hidden size that was used in training
    :param output_size: output size that was used in training
    :param dropout: value of dropout that was used in training (linear layer)
    """

    files = [join(seed_dir, f) for f in listdir(seed_dir)]
    model = MusicModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout=dropout,
                       batch_size=1)
    model.load_state_dict(torch.load(join(model_dir, 'model_weights.pth'), map_location=torch.device('cpu')))
    model.eval()

    if not isdir(save_dir):
        mkdir(save_dir)

    for i, file in enumerate(files, 1):
        model.init_hidden('cpu')

        sample_rate_song, data = wavfile.read(file)

        # data is only used as a seed. So the function is written in such a way that we can take a seed from any song
        # (although it's advisable to use songs with proper sample rate and number of channels)
        data = data.reshape(-1)
        seed = data[sample_rate_song * start_second:sample_rate_song * start_second + input_size // 2]

        no_samples = int(len_piece * sample_rate / (output_size // 2))
        song = np.empty(shape=(no_samples, output_size // 2), dtype=np.int16)

        x = torch.tensor(get_frequency_domains(seed).reshape(1, -1), dtype=torch.float)
        for j in range(no_samples):
            with torch.no_grad():
                # We use previous predictions to make new predictions.
                # We use int16 because that's the format in which all the songs in this dataset have been saved
                x = model(x)
                song[j] = generate_waveforms(x.numpy().reshape(-1)).real.astype(np.int16)

        # Save song
        song = song.reshape(-1)
        wavfile.write(join(save_dir, 'generated_song_' + str(i) + '.wav'), sample_rate, song)
