import numpy as np
from torch.utils.data import DataLoader
from dataloader.music_dataset import MusicDataset


def get_data_loader(x: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    """
    Generate a DataLoader from a given data

    :param x: input sequences
    :param y: output sequences
    :param batch_size: batch size
    :return: DataLoader
    """

    dataset = MusicDataset(x, y)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
    return dataloader
