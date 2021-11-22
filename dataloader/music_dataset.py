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

        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
