import os

import torch
from torch.utils import data
from torchvision import datasets, transforms


def split_train_val(full_train_data, test_data):
    val_len = len(test_data)
    train_len = len(full_train_data) - val_len
    train_data, val_data = data.random_split(full_train_data, [train_len, val_len])
    return train_data, val_data


def download_mnist(data_dir):
    """Downloading the fashion mnist data

    Args:
        data_dir (str): root directory

    Returns:

    """

    data_dir = data_dir or "./_data/mnist"
    os.makedirs(data_dir, exist_ok=True)

    download = not os.path.exists(os.path.join(data_dir, "MNIST", "raw"))

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
        target_transform=None,
    )
    train_data, val_data = split_train_val(full_train_data, test_data)

    return train_data, val_data, test_data


def download_fashion_mnist(data_dir):
    """Downloading the fashion mnist data

    Args:
        data_dir (str): root directory

    Returns:

    """

    data_dir = data_dir or "./_data/fashion_mnist"
    os.makedirs(data_dir, exist_ok=True)

    download = not os.path.exists(os.path.join(data_dir, "FashionMNIST", "raw"))

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
        target_transform=None,
    )

    train_data, val_data = split_train_val(full_train_data, test_data)
    return train_data, val_data, test_data


def download_cifar10(data_dir):
    """Downloading the fashion mnist data

    Args:
        data_dir (str): root directory

    Returns:

    """

    data_dir = data_dir or "./_data/cifar10"
    os.makedirs(data_dir, exist_ok=True)

    download = not os.path.exists(os.path.join(data_dir, "CIFAR10", "raw"))

    full_train_data = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=transforms.ToTensor(),
        target_transform=None,
    )

    test_data = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=transforms.ToTensor(),
        target_transform=None,
    )
    train_data, val_data = split_train_val(full_train_data, test_data)

    return train_data, val_data, test_data


def download_cifar100(data_dir):
    """Downloading the fashion mnist data

    Args:
        data_dir (str): root directory

    Returns:

    """

    data_dir = data_dir or "./_data/cifar100"
    os.makedirs(data_dir, exist_ok=True)

    download = not os.path.exists(os.path.join(data_dir, "CIFAR100", "raw"))

    full_train_data = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=download,
        transform=transforms.ToTensor(),
        target_transform=None,
    )

    test_data = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=download,
        transform=transforms.ToTensor(),
        target_transform=None,
    )
    train_data, val_data = split_train_val(full_train_data, test_data)

    return train_data, val_data, test_data


def download_project(project_name, data_dir):
    torch.manual_seed(41)

    download_fn = {
        "fashion_mnist": download_fashion_mnist,
        "mnist": download_mnist,
        "cifar10": download_cifar10,
        "cifar100": download_cifar100,
    }.get(project_name)

    train_data, val_data, test_data = download_fn(data_dir)
    return train_data, val_data, test_data
