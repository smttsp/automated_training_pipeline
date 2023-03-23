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


@hydra.main(version_base=None, config_path="./configs", config_name="fashion_mnist")
def main_fn(cfg):
    model = train_model(cfg)

    os.makedirs(cfg.training.save_dir, exist_ok=True)
    save_path = os.path.join(cfg.training.save_dir, f"{get_runtime_str()}.pth")
    torch.save(obj=model.state_dict(), f=save_path)
    return None


if __name__ == "__main__":
    main_fn()
