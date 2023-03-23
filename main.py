from training_pipeline.train import train_model

import hydra
from datetime import datetime


def get_runtime_str():
    """Getting datetime as a string"""
    runtime_str = (
        datetime.now().isoformat().replace(":", "").replace("-", "").replace("T", "-").split(".")[0]
    )
    return runtime_str


@hydra.main(version_base=None, config_path="./configs", config_name="fashion_mnist")
def main(cfg):
    model = train_model(cfg)

    save_dir = get_runtime_str()


if __name__ == "__main__":
    main()

