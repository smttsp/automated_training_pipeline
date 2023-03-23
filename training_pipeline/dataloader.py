from torch.utils.data import DataLoader

from training_pipeline.data_downloader import download_fashion_mnist


def get_dataloaders(batch_size=32):
    """Downloading the fashion mnist data

    Args:
        root (str): root directory

    Returns:

    """
    train_data, val_data, test_data = download_fashion_mnist()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
