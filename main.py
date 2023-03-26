import os
from datetime import datetime
import argparse
import hydra
import torch

from training_pipeline.train import train_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        default="mnist",
        required=False,  # it should be true by default and no default
        type=str,
        help="name of the config to be uses",
    )
    args = parser.parse_args()
    print(args.config_name)
    return args.config_name


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


config_name = parse_args()


@hydra.main(version_base=None, config_path="./configs", config_name=config_name)
def main_fn(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_model(cfg, device)
    save_model(cfg, model)

    return None


if __name__ == "__main__":
    main_fn()
