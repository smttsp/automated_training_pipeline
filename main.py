import os
import hydra
import torch

from training_pipeline.train import train_model
from training_pipeline.utils.io_utils import save_model, parse_args, get_runtime_str
from training_pipeline.utils.wandb_utils import create_wandb_project

PROJECT_NAME = parse_args()


@hydra.main(version_base=None, config_path="./configs", config_name=PROJECT_NAME)
def main_fn(cfg):
    runtime_str = get_runtime_str()
    wandb = create_wandb_project(PROJECT_NAME, runtime_str)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_model(cfg, device, wandb)
    save_model(cfg, model, runtime_str)

    wandb.finish()
    return None




if __name__ == "__main__":
    main_fn()
