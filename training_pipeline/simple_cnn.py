import torch
from torch import nn


class Simple_CNN_Classification(nn.Module):
    def __init__(self, input_shape: torch.Size, hidden_units: int, output_shape: int):
        super().__init__()

        input_channels = input_shape[1]
        input_hw = input_shape[-1]
        num_conv_block = 2

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        in_features_for_flat_layer = int(hidden_units * (input_hw / 2 ** num_conv_block) ** 2)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features_for_flat_layer, out_features=output_shape),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
