from training_pipeline.train import train_model

import hydra


@hydra.main(version_base=None, config_path="./configs", config_name="fashion_mnist")
def main(cfg):

    train_model(cfg)


if __name__ == "__main__":
    main()

