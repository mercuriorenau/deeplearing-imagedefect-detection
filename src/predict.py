"""
Predict defect / no_defect for one or more images (or a folder).
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import DefectClassifier


def load_config(config_path="config.yaml"):
    with open(Path(ROOT) / config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_inference_transform(image_size=224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


def load_model(checkpoint_path, config_path="config.yaml", device=None):
    cfg = load_config(config_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DefectClassifier(
        backbone_name=cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=False,
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    class_to_idx = ckpt.get("class_to_idx", {"no_defect": 0, "defect": 1})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return model, idx_to_class, cfg["image"]["size"]


def predict_image(model, image_path, transform, device, idx_to_class):
    img = Image.open(image_path).convert("RGB")
    return predict_image_from_pil(model, img, transform, device, idx_to_class)


def predict_image_from_pil(model, pil_image, transform, device, idx_to_class):
    """Predict from a PIL Image (e.g. for web upload)."""
    img = pil_image.convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = logits.argmax(dim=1).item()
    label = idx_to_class.get(pred_idx, str(pred_idx))
    conf = probs[0][pred_idx].item()
    return label, conf


def main():
    parser = argparse.ArgumentParser(description="Predict defect / no_defect for image(s)")
    parser.add_argument("input", nargs="+", help="Image file(s) or folder with images")
    parser.add_argument("--checkpoint", default=None, help="Path to .pt (default: from config)")
    parser.add_argument("--config", default="config.yaml", help="Config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = args.checkpoint or str(Path(ROOT) / cfg["output"]["best_model"])
    if not Path(checkpoint).exists():
        print(f"Checkpoint not found: {checkpoint}. Train first with: python -m src.train")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idx_to_class, image_size = load_model(checkpoint, args.config, device)
    transform = get_inference_transform(image_size)

    inputs = []
    for p in args.input:
        path = Path(p)
        if path.is_file():
            if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                inputs.append(path)
        elif path.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                inputs.extend(path.glob(ext))

    if not inputs:
        print("No images found.")
        sys.exit(1)

    print(f"Classes: {idx_to_class}")
    print("-" * 50)
    for img_path in inputs:
        label, conf = predict_image(model, img_path, transform, device, idx_to_class)
        print(f"{img_path.name}: {label} ({conf:.2%})")


if __name__ == "__main__":
    main()
