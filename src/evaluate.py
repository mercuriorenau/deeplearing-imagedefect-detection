"""
Evaluate saved checkpoints on the validation / test split.
Reports accuracy, precision, recall, F1, confusion matrix, and majority baseline.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import get_mvtec_categories, get_mvtec_dataloaders
from src.metrics_utils import binary_classification_metrics
from src.model import DefectClassifier


def load_config(config_path="config.yaml"):
    with open(Path(ROOT) / config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def majority_baseline(val_dataset):
    """Accuracy if we always predict the most common class in the val split."""
    labels = [label for _, label in val_dataset.samples]
    if not labels:
        return 0.0, 0
    counts = Counter(labels)
    return max(counts.values()) / len(labels), len(labels)


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Return lists of true labels and predicted labels."""
    model.eval()
    y_true = []
    y_pred = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().tolist()
        y_true.extend(labels.tolist())
        y_pred.extend(preds)
    return y_true, y_pred


def evaluate_category(cfg, category, mvtec_root, device):
    category_dir = mvtec_root / category
    checkpoint_path = Path(ROOT) / cfg["output"]["checkpoint_dir"] / f"best_model_{category}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    batch_size = cfg["training"]["batch_size"]
    image_size = cfg["image"]["size"]
    num_workers = cfg["training"]["num_workers"]
    train_ratio = cfg["data"].get("train_ratio", 0.8)
    split_mode = cfg["data"].get("split_mode", "pooled_random")

    _, val_loader, class_to_idx = get_mvtec_dataloaders(
        str(category_dir),
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        train_ratio=train_ratio,
        split_mode=split_mode,
    )

    model = DefectClassifier(
        backbone_name=cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=False,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    y_true, y_pred = collect_predictions(model, val_loader, device)
    metrics = binary_classification_metrics(y_true, y_pred, positive_label=1)
    baseline, n_val = majority_baseline(val_loader.dataset)

    return {
        "category": category,
        "val_accuracy": round(metrics["accuracy"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1": round(metrics["f1"], 4),
        "confusion_matrix": metrics["confusion_matrix"],
        "majority_baseline": round(baseline, 4),
        "improvement_vs_baseline": round(metrics["accuracy"] - baseline, 4),
        "n_val_samples": n_val,
        "class_to_idx": ckpt.get("class_to_idx", class_to_idx),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
    }


def _split_description(cfg):
    split_mode = cfg["data"].get("split_mode", "pooled_random")
    if split_mode == "official_holdout":
        return "official_holdout (train on train/good + train-ratio of test defects; eval on held-out test)"
    return "validation (random 80/20 from MVTec train+test pool)"


def main():
    parser = argparse.ArgumentParser(description="Evaluate defect classifiers on validation split")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--category", default=None, help="MVTec category (e.g. bottle)")
    parser.add_argument("--all-categories", action="store_true", help="Evaluate all categories with checkpoints")
    parser.add_argument("--output", default="results/metrics.json", help="Output JSON path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mvtec_root = Path(cfg["data"].get("mvtec_root", "data"))
    if not mvtec_root.is_absolute():
        mvtec_root = Path(ROOT) / mvtec_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.all_categories:
        categories = get_mvtec_categories(mvtec_root)
        ckpt_dir = Path(ROOT) / cfg["output"]["checkpoint_dir"]
        categories = [
            c for c in categories
            if (ckpt_dir / f"best_model_{c}.pt").exists()
        ]
        if not categories:
            print("No checkpoints found for MVTec categories.")
            sys.exit(1)
    elif args.category:
        categories = [args.category]
    else:
        category = cfg["data"].get("mvtec_category")
        if not category:
            print("Specify --category or --all-categories")
            sys.exit(1)
        categories = [category]

    results = []
    for cat in categories:
        print(f"\n--- {cat} ---")
        try:
            metrics = evaluate_category(cfg, cat, mvtec_root, device)
            results.append(metrics)
            print(
                f"Val accuracy: {metrics['val_accuracy']:.2%} | "
                f"P/R/F1: {metrics['precision']:.2%}/{metrics['recall']:.2%}/{metrics['f1']:.2%} | "
                f"Baseline: {metrics['majority_baseline']:.2%} | "
                f"Improvement: {metrics['improvement_vs_baseline']:+.2%} | "
                f"n={metrics['n_val_samples']}"
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipped {cat}: {exc}")

    if not results:
        print("No categories evaluated.")
        sys.exit(1)

    output_path = Path(ROOT) / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": _split_description(cfg),
        "split_mode": cfg["data"].get("split_mode", "pooled_random"),
        "model": cfg["model"]["name"],
        "categories": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
