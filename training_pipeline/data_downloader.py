import os
from torchvision import datasets, transforms
import torch
from torch.utils import data


def download_mnist(data_dir):
    """Downloading the fashion mnist data

    Args:
        data_dir (str): root directory

    Returns:

    """
    torch.manual_seed(41)

    data_dir = data_dir or "./_data/mnist"

    os.makedirs(data_dir, exist_ok=True)

    # if data already exists, don't download it. Pull from the disk
    download = not os.path.exists(os.path.join(data_dir, 'MNIST', 'raw'))

    full_train_data = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transforms.ToTensor(),
        target_transform=None,
    )

    test_data = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transforms.ToTensor(),
        target_transform=None
    )
    val_len = len(test_data)
    train_len = len(full_train_data)-val_len

    train_data, val_data = data.random_split(full_train_data, [train_len, val_len])

    return train_data, val_data, test_data


def download_fashion_mnist(data_dir):
    """Downloading the fashion mnist data

    Args:
        data_dir (str): root directory

    Returns:

    """

    data_dir = data_dir or "./_data/fashion_mnist"

    torch.manual_seed(41)
    os.makedirs(data_dir, exist_ok=True)

    # if data already exists, don't download it. Pull from the disk
    download = not os.path.exists(os.path.join(data_dir, 'FashionMNIST', 'raw'))

    full_train_data = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transforms.ToTensor(),
        target_transform=None,
    )

    test_data = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transforms.ToTensor(),
        target_transform=None
    )
    val_len = len(test_data)
    train_len = len(full_train_data)-val_len

    train_data, val_data = data.random_split(full_train_data, [train_len, val_len])

    return train_data, val_data, test_data
