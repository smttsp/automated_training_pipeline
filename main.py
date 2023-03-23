from training_pipeline.dataloader import download_fashion_mnist


if __name__ == "__main__":
    download_fashion_mnist("_data/fashion_mnist/")


    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
