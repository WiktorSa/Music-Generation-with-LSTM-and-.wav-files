import torch
from tqdm import tqdm


def test(model, test_dataloader, criterion, device: str) -> None:
    """
    Test a model

    :param model: model to train on
    :param test_dataloader: test DataLoader
    :param criterion: criterion
    :param device: the device to use in calculations. Either 'cpu' or 'cuda'
    """

    model.eval()
    test_loss = 0
    no_test_inputs = 0

    test_bar = tqdm(test_dataloader, total=len(test_dataloader), desc=f'Test')
    for x_seq, y_seq in test_bar:
        x_seq = x_seq.to(device)
        y_seq = y_seq.to(device)

        model.init_hidden(device)
        with torch.no_grad():
            y_pred = model(x_seq)
            loss = criterion(y_pred, y_seq)

        test_loss += loss.item()
        no_test_inputs += len(y_seq)

        test_bar.set_postfix_str(f'Test loss: {test_loss / no_test_inputs:.4f}')
