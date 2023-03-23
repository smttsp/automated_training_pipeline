import os
from torchvision import datasets, transforms


def download_fashion_mnist(root="_data", download=True):
    """Downloading the fashion mnist data

    Args:
        root ():
        download ():

    Returns:

    """

    os.makedirs(root, exist_ok=True)

    train_data = datasets.FashionMNIST(
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
    return train_data, test_data



def load_train_test_data():
    pass
