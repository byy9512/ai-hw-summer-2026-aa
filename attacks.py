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


def ifgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float = 0.01,
    num_iter: int = 40,
    random_start: bool = False,
) -> torch.Tensor:
    """Iterative FGSM (Kurakin et al., 2017) / Projected Gradient Descent (Madry et al., 2018).

    Repeats small FGSM steps, projecting back into the L-infinity epsilon-ball around the
    original image after each step (and clamping to valid pixel range). PGD is this same
    procedure starting from a random point inside that ball (random_start=True); I-FGSM
    starts exactly at the original image (random_start=False) — otherwise identical.
    """
    original = images.clone().detach()

    if random_start:
        perturbed = original + torch.empty_like(original).uniform_(-epsilon, epsilon)
        perturbed = perturbed.clamp(0, 1).detach()
    else:
        perturbed = original.clone().detach()

    for _ in range(num_iter):
        perturbed.requires_grad_(True)
        outputs = model(perturbed)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            perturbed = perturbed + alpha * perturbed.grad.sign()
            perturbation = (perturbed - original).clamp(-epsilon, epsilon)
            perturbed = (original + perturbation).clamp(0, 1)
        perturbed = perturbed.detach()

    return perturbed


def mifgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float = 0.01,
    num_iter: int = 40,
    mu: float = 1.0,
) -> torch.Tensor:
    """Momentum Iterative FGSM (Dong et al., 2018).

    Same iterative structure as I-FGSM, but accumulates a momentum term over the
    (L1-normalized) gradient direction at each step before taking the sign step.
    Momentum stabilizes the update direction and helps escape poor local maxima of
    the loss surface. Uses the same alpha/num_iter as ifgsm_attack so momentum is
    the only variable being tested between the two.
    """
    original = images.clone().detach()
    perturbed = original.clone().detach()
    momentum = torch.zeros_like(original)

    for _ in range(num_iter):
        perturbed.requires_grad_(True)
        outputs = model(perturbed)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()

        grad = perturbed.grad
        grad_norm = grad.abs().sum(dim=(1, 2, 3), keepdim=True).clamp(min=1e-12)
        momentum = mu * momentum + grad / grad_norm

        with torch.no_grad():
            perturbed = perturbed + alpha * momentum.sign()
            perturbation = (perturbed - original).clamp(-epsilon, epsilon)
            perturbed = (original + perturbation).clamp(0, 1)
        perturbed = perturbed.detach()

    return perturbed
