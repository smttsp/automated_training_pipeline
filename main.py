import os
from datetime import datetime
import argparse
import hydra
import torch

from training_pipeline.train import train_model
from training_pipeline.utils.io_utils import save_model, parse_args, get_runtime_str
# from training_pipeline.utils.neptune_utils import create_or_select_project

CONFIG_NAME = parse_args()
NEPTUNE_API_KEY = os.environ["NEPTUNE_API_KEY"]


@hydra.main(version_base=None, config_path="./configs", config_name=CONFIG_NAME)
def main_fn(cfg):
    runtime_str = get_runtime_str()
    # create_or_select_project(CONFIG_NAME, NEPTUNE_API_KEY)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_model(cfg, device)
    save_model(cfg, model, runtime_str)

    return None


if __name__ == "__main__":
    main_fn()
