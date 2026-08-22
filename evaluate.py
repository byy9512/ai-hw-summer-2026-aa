import argparse
from pathlib import Path

import torch

from attacks import NormalizedModel, fgsm_attack, get_device, ifgsm_attack, mifgsm_attack
from data import get_raw_test_loader
from registry import CHECKPOINTS, MODEL_REGISTRY

CHECKPOINT_DIR = Path("checkpoints")

ATTACKS = {
    "fgsm": fgsm_attack,
    "ifgsm": lambda model, images, labels, epsilon: ifgsm_attack(model, images, labels, epsilon, random_start=False),
    "pgd": lambda model, images, labels, epsilon: ifgsm_attack(model, images, labels, epsilon, random_start=True),
    "mifgsm": mifgsm_attack,
}


def load_model(arch_key: str, ckpt_name: str, device: torch.device) -> NormalizedModel:
    model = MODEL_REGISTRY[arch_key]()
    model.load_state_dict(torch.load(CHECKPOINT_DIR / f"{ckpt_name}.pt", map_location=device))
    model.eval()
    return NormalizedModel(model).to(device)


def evaluate(model: NormalizedModel, test_loader, device: torch.device, attack_fn, epsilons: list[float]) -> dict:
    """Returns recognition rate and, for each epsilon, the attack success rate (ASR) —
    the fraction of originally-correctly-classified test images the attack flips to wrong.
    Images the model already misclassifies are excluded from the ASR denominator, since
    "fooling" an already-wrong prediction isn't a meaningful attack success.
    """
    correct = 0
    total = 0
    post_attack_correct = {eps: 0 for eps in epsilons}
    attacked_count = {eps: 0 for eps in epsilons}

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            preds = model(images).argmax(dim=1)
        correct_mask = preds == labels
        correct += correct_mask.sum().item()
        total += labels.size(0)

        if correct_mask.sum().item() == 0:
            continue

        target_images = images[correct_mask]
        target_labels = labels[correct_mask]

        for eps in epsilons:
            perturbed = attack_fn(model, target_images, target_labels, eps)
            with torch.no_grad():
                adv_preds = model(perturbed).argmax(dim=1)
            post_attack_correct[eps] += (adv_preds == target_labels).sum().item()
            attacked_count[eps] += target_labels.size(0)

    results = {"recognition_rate": 100 * correct / total}
    for eps in epsilons:
        asr = 100 * (1 - post_attack_correct[eps] / attacked_count[eps])
        results[eps] = asr
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate an attack against a frozen checkpoint")
    parser.add_argument("checkpoint", choices=[c for c, _ in CHECKPOINTS])
    parser.add_argument("--attack", choices=ATTACKS.keys(), default="fgsm")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    arch_key = dict(CHECKPOINTS)[args.checkpoint]
    device = get_device()
    print(f"using device: {device}")

    test_loader = get_raw_test_loader(batch_size=args.batch_size)
    model = load_model(arch_key, args.checkpoint, device)

    results = evaluate(model, test_loader, device, ATTACKS[args.attack], args.epsilons)

    print(f"\n{args.checkpoint} ({args.attack})")
    print(f"  recognition rate: {results['recognition_rate']:.2f}%")
    for eps in args.epsilons:
        print(f"  epsilon={eps:.2f}  ASR={results[eps]:.2f}%")


if __name__ == "__main__":
    main()
