from torch.utils.data import DataLoader

from training_pipeline.data_downloader import download_fashion_mnist, download_project


def get_dataloaders(cfg):
    """Downloading the fashion mnist data

    Args:
        root (str): root directory

    Returns:

    """

    batch_size = cfg.get("training", {}).get("batch_size", 32)
    project_name = cfg.get("project_name")
    data_dir = cfg.get("data_dir")

    train_data, val_data, test_data = download_project(project_name, data_dir)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
