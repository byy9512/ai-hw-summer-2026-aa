import torch.nn as nn


class MLP(nn.Module):
    """Shallow MLP for MNIST digit classification (28x28 -> 10 classes)."""

    def __init__(self, input_size: int = 28 * 28, hidden_size: int = 128, num_classes: int = 10, dropout: float = 0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)
