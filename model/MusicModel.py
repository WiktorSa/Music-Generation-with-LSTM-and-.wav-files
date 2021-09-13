import torch
import torch.nn as nn


class MusicModel(nn.Module):
    def __init__(self, input_size: int = 8000, hidden_size: int = 2048, output_size: int = 4000, dropout: float = 0.2,
                 batch_size: int = 40):
        """
        Build a model made for training on music data.
        Model consists of one fully connected layer and one LSTM

        :param input_size: size of the input data
        :param hidden_size: hidden size
        :param output_size: size of the output data
        :param dropout: value of dropout for fully connected layer
        :param no_timesteps: the number of timesteps to perform in one iteration of the model
        """

        super(MusicModel, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        # User needs to call init_hidden firstly for safety reasons
        self.hidden = None

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        lstm_out, self.hidden = self.lstm(x.view(len(x), -1, 1), self.hidden)
        prediction = self.linear2(lstm_out)
        return prediction[:, -1]

    def init_hidden(self, device):
        self.hidden = (torch.zeros(1, self.batch_size, self.hidden_size).to(device),
                       torch.zeros(1, self.batch_size, self.hidden_size).to(device))
