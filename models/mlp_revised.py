import torch.nn as nn


class MLPRevised(nn.Module):
    """MLP with BatchNorm and a second hidden layer for added capacity."""

    def __init__(self, input_size: int = 28 * 28, hidden_sizes=(256, 128), num_classes: int = 10, dropout: float = 0.2):
        super().__init__()
        h1, h2 = hidden_sizes
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_size, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(h2, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)
