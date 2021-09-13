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
    :param device: the device to use in calculations. Either 'cpu' or 'gpu'
    :param no_epochs: the number of epochs to train the model
    """

    for epoch in range(no_epochs):
        # Training
        model.train()
        train_loss = 0
        no_train_inputs = 0

        train_bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f'Train on epoch {epoch}')
        for x_seq, y_seq in train_bar:
            x_seq.to(device)
            y_seq.to(device)

            optimizer.zero_grad()
            model.init_hidden()
            y_pred = model(x_seq)
            loss = criterion(y_pred, y_seq)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            no_train_inputs += len(y_seq)

            train_bar.set_postfix_str(f'Train loss: {train_loss / no_train_inputs:.4f}')

        # Validation
        model.eval()
        val_loss = 0
        no_val_inputs = 0

        val_bar = tqdm(val_dataloader, total=len(val_dataloader), desc=f'Validation on epoch {epoch}')
        for x_seq, y_seq in val_bar:
            x_seq.to(device)
            y_seq.to(device)

            model.init_hidden()
            with torch.no_grad():
                y_pred = model(x_seq)
                loss = criterion(y_pred, y_seq)

            val_loss += loss.item()
            no_val_inputs += len(y_seq)

            val_bar.set_postfix_str(f'Validation loss: {val_loss / no_val_inputs:.4f}')
