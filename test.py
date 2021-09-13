import numpy as np
import torch
import torch.nn as nn
from dataloader import get_data_loader
from model import MusicModel
from utils import print_model_info, train, test

# The code below should be deleted
from dataloader.MusicDataset import MusicDataset
from torch.utils.data import DataLoader
x = np.zeros(shape=(1, 1, 8000)).astype(np.float32)
y = np.zeros(shape=(1, 1, 4000)).astype(np.float32)
train_dataloader = get_data_loader(x, y)
val_dataloader = get_data_loader(x, y)
test_dataloader = get_data_loader(x, y)

data = np.load('data/data.npz')

# train_dataloader = get_data_loader(data['X_train'], data['y_train'])
# val_dataloader = get_data_loader(data['X_val'], data['y_val'])
# test_dataloader = get_data_loader(data['X_test'], data['y_test'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MusicModel(no_timesteps=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-4)

# train(model, train_dataloader, val_dataloader, criterion, optimizer, device)
test(model, test_dataloader, criterion, device)
