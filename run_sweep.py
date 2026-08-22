import json
from pathlib import Path

import torch

from attacks import get_device
from data import get_raw_test_loader
from evaluate import ATTACKS, load_model
from registry import CHECKPOINTS

RESULTS_PATH = Path("results.json")
EPSILONS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def evaluate_checkpoint(model, test_loader, device) -> dict:
    """Computes recognition rate once, then every attack x epsilon combo on the images the
    model originally got right (see evaluate.evaluate for the ASR definition)."""
    correct = 0
    total = 0
    post_attack_correct = {name: {eps: 0 for eps in EPSILONS} for name in ATTACKS}
    attacked_count = {name: {eps: 0 for eps in EPSILONS} for name in ATTACKS}

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

        for attack_name, attack_fn in ATTACKS.items():
            for eps in EPSILONS:
                perturbed = attack_fn(model, target_images, target_labels, eps)
                with torch.no_grad():
                    adv_preds = model(perturbed).argmax(dim=1)
                post_attack_correct[attack_name][eps] += (adv_preds == target_labels).sum().item()
                attacked_count[attack_name][eps] += target_labels.size(0)

    results = {"recognition_rate": 100 * correct / total, "attacks": {}}
    for attack_name in ATTACKS:
        results["attacks"][attack_name] = {
            str(eps): 100 * (1 - post_attack_correct[attack_name][eps] / attacked_count[attack_name][eps])
            for eps in EPSILONS
        }
    return results


def main():
    device = get_device()
    print(f"using device: {device}")

    test_loader = get_raw_test_loader(batch_size=256)

    all_results = {}
    if RESULTS_PATH.exists():
        all_results = json.loads(RESULTS_PATH.read_text())

    for ckpt_name, arch_key in CHECKPOINTS:
        if ckpt_name in all_results:
            print(f"skipping {ckpt_name} (already in {RESULTS_PATH})")
            continue

        model = load_model(arch_key, ckpt_name, device)
        results = evaluate_checkpoint(model, test_loader, device)

        all_results[ckpt_name] = results
        RESULTS_PATH.write_text(json.dumps(all_results, indent=2))

        print(f"{ckpt_name}: recognition_rate={results['recognition_rate']:.2f}%")
        for attack_name, per_eps in results["attacks"].items():
            summary = "  ".join(f"{eps}={asr:.1f}%" for eps, asr in per_eps.items())
            print(f"  {attack_name}: {summary}")


if __name__ == "__main__":
    main()
