import numpy as np
from dataloader import get_data_loader
from model import MusicModel

data = np.load('data/data.npz')

train_dataloader = get_data_loader(data['X_train'], data['y_train'])
val_dataloader = get_data_loader(data['X_val'], data['y_val'])
test_dataloader = get_data_loader(data['X_test'], data['y_test'])

model = MusicModel()

for X_train, y_train in train_dataloader:
    prediction = model(X_train)
    print(prediction.shape)
    break
