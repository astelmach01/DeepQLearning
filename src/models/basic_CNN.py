import torch
from torch import nn


def get_model(in_channels: int, output_dim: int) -> torch.nn.Sequential:
    online = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=64,
                  kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=32,
                  kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32,
                  kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3168, 512),
        nn.ReLU(),
        nn.Linear(512, output_dim),
    )

    return online
