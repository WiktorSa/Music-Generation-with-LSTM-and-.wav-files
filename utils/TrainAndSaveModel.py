import numpy as np
from os import mkdir
from os.path import join, isdir
from dataloader import get_data_loader
from model import MusicModel
from utils.Train import train
from utils.Test import test


def train_and_save_model(directory: str):
    data = np.load(join(directory, 'data.npz'))

    train_dataloader = get_data_loader(data['X_train'], data['y_train'])
    val_dataloader = get_data_loader(data['X_val'], data['y_val'])
    test_dataloader = get_data_loader(data['X_test'], data['y_test'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.
    model = MusicModel(input_size=1, hidden_size=1, output_size=1, dropout=1, batch_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1, weight_decay=1)

    # train(model, train_dataloader, val_dataloader, criterion, optimizer, device)
    # test(model, test_dataloader, criterion, device)


