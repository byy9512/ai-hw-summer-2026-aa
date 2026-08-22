import torch
import torch.nn as nn
import torch.nn.functional as F

from data import MNIST_MEAN, MNIST_STD


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NormalizedModel(nn.Module):
    """Wraps a model trained on normalized MNIST so attacks can operate directly in raw
    [0, 1] pixel space — perturbation budgets and clipping are meaningful in that space,
    not in the shifted/scaled space the model was actually trained on."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MNIST_MEAN).view(1, 1, 1, 1))
        self.register_buffer("std", torch.tensor(MNIST_STD).view(1, 1, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def fgsm_attack(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Fast Gradient Sign Method (Goodfellow et al., 2015).

    Single step in the direction that most increases the loss: x_adv = x + eps * sign(grad_x loss).
    images must be in raw [0, 1] pixel space; the result is clamped back into that range.
    """
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()

    perturbed = images + epsilon * images.grad.sign()
    return perturbed.clamp(0, 1).detach()
