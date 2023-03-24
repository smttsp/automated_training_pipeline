import os
from datetime import datetime

import hydra
import torch

from training_pipeline.train import train_model


def get_runtime_str():
    """Getting datetime as a string"""
    runtime_str = (
        datetime.now().isoformat().replace(":", "").replace("-", "").replace("T", "-").split(".")[0]
    )
    return runtime_str


def save_model(cfg, model):
    project_name = cfg.project_name
    save_dir = cfg.training.save_dir

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{project_name}_{get_runtime_str()}.pth")
    torch.save(obj=model, f=save_path)


@hydra.main(version_base=None, config_path="./configs", config_name="fashion_mnist")
def main_fn(cfg):
    model = train_model(cfg)
    save_model(cfg, model)

    return None


if __name__ == "__main__":
    main_fn()
