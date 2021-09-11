import numpy as np
from torch.utils.data import DataLoader
from dataloader.MusicDataset import MusicDataset


def get_data_loader(X: np.ndarray, y: np.ndarray) -> DataLoader:
    """
    Generate a DataLoader from a given data

    :param X: input sequences
    :param y: output sequences
    :return: DataLoader
    """

    batch_size = X.shape[1]
    dataset = MusicDataset(X, y)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    return dataloader
