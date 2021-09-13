import numpy as np
from torch.utils.data import DataLoader
from dataloader.MusicDataset import MusicDataset


def get_data_loader(x: np.ndarray, y: np.ndarray) -> DataLoader:
    """
    Generate a DataLoader from a given data

    :param x: input sequences
    :param y: output sequences
    :return: DataLoader
    """

    batch_size = x.shape[1]
    dataset = MusicDataset(x, y)
    dataloader = DataLoader(dataset, batch_size)
    return dataloader
