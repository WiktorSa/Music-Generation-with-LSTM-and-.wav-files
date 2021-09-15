import numpy as np
import torch
import torch.nn as nn
from os import mkdir
from os.path import join, isdir
from dataloader import get_data_loader
from model import MusicModel
from utils.Train import train
from utils.Test import test


def set_seed(seed: int):
    """
    Set a seed for torch

    :param seed: seed
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_and_save_model(directory: str = 'data', batch_size: int = 40, input_size: int = 8000,
                         hidden_size: int = 2048, output_size: int = 8000, dropout: float = 0.2, lr: float = 1e-4,
                         weight_decay: float = 1e-4, no_epochs: int = 2000, save_model: str = 'model_weights',
                         seed: int = 1001) -> None:
    """
    Train and save model

    :param directory: directory with preprocessed data
    :param input_size: size of the input
    :param hidden_size: hidden size
    :param output_size: size of the output
    :param dropout: value of dropout (linear layer)
    :param batch_size: batch size
    :param lr: learning rate (optimizer)
    :param weight_decay: value of weight decay (optimizer)
    :param no_epochs: number of epochs
    :param save_model: directory where model should be saved
    :param seed: seed
    """

    data = np.load(join(directory, 'data.npz'))

    train_dataloader = get_data_loader(data['X_train'], data['y_train'], batch_size)
    val_dataloader = get_data_loader(data['X_val'], data['y_val'], batch_size)
    test_dataloader = get_data_loader(data['X_test'], data['y_test'], batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seed(seed)
    model = MusicModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout=dropout,
                       batch_size=batch_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    train(model, train_dataloader, val_dataloader, criterion, optimizer, device, no_epochs)
    test(model, test_dataloader, criterion, device)

    # Save model weights
    if not isdir(save_model):
        mkdir(save_model)

    torch.save(model.state_dict(), join(save_model, 'model_weights.pth'))
