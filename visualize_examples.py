import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from attacks import fgsm_attack, get_device, ifgsm_attack, mifgsm_attack
from data import get_raw_test_loader
from evaluate import load_model

CHECKPOINT = "cnn_revised_aug"
ARCH = "cnn_revised"
EPSILON = 0.15
NUM_DIGITS = 6

ATTACKS = [
    ("FGSM", fgsm_attack),
    ("PGD", lambda m, x, y, e: ifgsm_attack(m, x, y, e, random_start=True)),
    ("MI-FGSM", mifgsm_attack),
]


def main():
    device = get_device()
    model = load_model(ARCH, CHECKPOINT, device)
    test_loader = get_raw_test_loader(batch_size=256)

    # collect a handful of correctly-classified, distinct-digit examples
    picked_images, picked_labels = [], []
    seen_digits = set()
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            preds = model(images).argmax(dim=1)
        for img, label, pred in zip(images, labels, preds):
            digit = label.item()
            if pred.item() == digit and digit not in seen_digits:
                picked_images.append(img)
                picked_labels.append(digit)
                seen_digits.add(digit)
            if len(picked_images) >= NUM_DIGITS:
                break
        if len(picked_images) >= NUM_DIGITS:
            break

    images = torch.stack(picked_images)
    labels = torch.tensor(picked_labels, device=device)

    columns = [("Original", None)] + ATTACKS
    fig, axes = plt.subplots(NUM_DIGITS, len(columns), figsize=(2.2 * len(columns), 2.2 * NUM_DIGITS))

    for row in range(NUM_DIGITS):
        img = images[row : row + 1]
        label = labels[row : row + 1]

        for col, (name, attack_fn) in enumerate(columns):
            shown = img if attack_fn is None else attack_fn(model, img, label, EPSILON)
            with torch.no_grad():
                pred = model(shown).argmax(dim=1).item()

            ax = axes[row, col]
            ax.imshow(shown[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")

            correct = pred == label.item()
            color = "#0b7a0b" if correct else "#c0392b"
            title = f"true={label.item()}, pred={pred}"
            if row == 0:
                title = f"{name}\n{title}"
            ax.set_title(title, fontsize=9, color=color)

    fig.suptitle(f"Original vs. adversarial examples — {CHECKPOINT} (eps={EPSILON})", fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig("examples.png", dpi=150, facecolor="white")
    print("wrote examples.png")


if __name__ == "__main__":
    main()
