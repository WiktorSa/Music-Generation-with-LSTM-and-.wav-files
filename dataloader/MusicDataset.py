import numpy as np
import torch
from torch.utils.data import Dataset


class MusicDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Create a music dataset for a given data

        :param x: input sequences
        :param y: output sequences
        """

        self.x = x.reshape(-1, x.shape[2])
        self.y = y.reshape(-1, y.shape[2])

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), self.y