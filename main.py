from training_pipeline.train import train_model
import os
import hydra
from datetime import datetime
import torch


def get_runtime_str():
    """Getting datetime as a string"""
    runtime_str = (
        datetime.now().isoformat().replace(":", "").replace("-", "").replace("T", "-").split(".")[0]
    )
    return runtime_str


@hydra.main(version_base=None, config_path="./configs", config_name="fashion_mnist")
def main_fn(cfg):
    model = train_model(cfg)
    save_path = os.path.join(cfg.training.save_dir, f"{get_runtime_str()}.pth")
    torch.save(obj=model.state_dict(), f=save_path)
    return None


if __name__ == "__main__":
    main_fn()
