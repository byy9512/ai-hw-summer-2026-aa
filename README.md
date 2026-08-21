# ai-hw-summer-2026-aa

Adversarial attacks on an MNIST digit classifier — fooling a trained model and measuring how well it holds up.

## Data

[MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) via `torchvision.datasets.MNIST` (mirrors [huggingface.co/datasets/ylecun/mnist](https://huggingface.co/datasets/ylecun/mnist)). Trained on the train split, evaluated on the test split.

## Attacks

- **FGSM** — Fast Gradient Sign Method
- **I-FGSM / PGD** — Iterative FGSM / Projected Gradient Descent
- **MI-FGSM** — Momentum Iterative FGSM

## Metrics

- **Recognition rate before attacks** — clean test accuracy of the target model
- **Attack success rate (ASR)** — how often each attack causes a misclassification

