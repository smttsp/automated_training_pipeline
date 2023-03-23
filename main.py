from training_pipeline.train import train_model

import hydra


@hydra.main(version_base="1.1", config_path="./configs", config_name="fashion_mnist")
def main(cfg):
    train_model(epochs=4)


if __name__ == "__main__":
    main()

