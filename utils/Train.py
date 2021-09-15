import torch
from tqdm import tqdm


def train(model, train_dataloader, val_dataloader, criterion, optimizer, device: str, no_epochs: int = 2000) -> None:
    """
    Train a model and validate it's performance

    :param model: model to train on
    :param train_dataloader: training DataLoader
    :param val_dataloader: validation DataLoader
    :param criterion: criterion
    :param optimizer: optimizer
    :param device: the device to use in calculations. Either 'cpu' or 'cuda'
    :param no_epochs: the number of epochs to train the model
    """

    for epoch in range(no_epochs):
        # Training
        model.train()
        train_loss = 0=

        train_bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f'Train epoch {epoch+1}')
        for x_seq, y_seq in train_bar:
            x_seq = x_seq.to(device)
            y_seq = y_seq.to(device)

            optimizer.zero_grad()
            model.init_hidden(device)
            y_pred = model(x_seq)
            loss = criterion(y_pred, y_seq)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            train_bar.set_postfix_str(f'Train loss: {train_loss:.6f}')

        # Validation
        model.eval()
        val_loss = 0
        val_bar = tqdm(val_dataloader, total=len(val_dataloader), desc=f'Val epoch {epoch+1}')
        for x_seq, y_seq in val_bar:
            x_seq = x_seq.to(device)
            y_seq = y_seq.to(device)

            model.init_hidden(device)
            with torch.no_grad():
                y_pred = model(x_seq)
                loss = criterion(y_pred, y_seq)

            val_loss += loss.item()
            val_bar.set_postfix_str(f'Val loss: {val_loss:.6f}')
