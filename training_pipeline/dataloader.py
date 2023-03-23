import os
from torchvision import datasets, transforms
import torch
from torch.utils import data


def download_fashion_mnist(root="_data/fashion_mnist"):
    """Downloading the fashion mnist data

    Args:
        root (str): root directory

    Returns:

    """
    torch.manual_seed(41)
    os.makedirs(root, exist_ok=True)

    # if data already exists, don't download it. Pull from the disk
    download = not os.path.exists(os.path.join(root, 'FashionMNIST', 'processed'))

    full_train_data = datasets.FashionMNIST(
        root=root,
        train=True,
        download=download,
        transform=transforms.ToTensor(),
        target_transform=None,
    )

    test_data = datasets.FashionMNIST(
        root=root,
        train=False,
        download=download,
        transform=transforms.ToTensor(),
        target_transform=None
    )
    val_len = len(test_data)
    train_len = len(full_train_data)-val_len

    train_data, val_data = data.random_split(full_train_data, [train_len, val_len])

    return train_data, val_data, test_data
