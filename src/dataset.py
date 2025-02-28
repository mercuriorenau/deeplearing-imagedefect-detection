"""
Dataset and DataLoaders for defect / no_defect classification.
Supports: (1) MVTec AD layout (category with train/, test/good/, test/<defect_types>/),
          (2) One folder with defect/ and no_defect/ (auto train/val split),
          (3) Separate train_dir and val_dir.
"""

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image


# Class names to index (consistent across MVTec and custom folders)
CLASS_TO_IDX = {"no_defect": 0, "defect": 1}
IDX_TO_CLASS = {0: "no_defect", 1: "defect"}

# Image extensions for MVTec-style folders
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def get_transforms(image_size=224, is_training=True):
    """Image transforms for train or val."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


def _apply_class_mapping(dataset):
    """Remap ImageFolder labels to CLASS_TO_IDX (no_defect=0, defect=1)."""
    dataset.class_to_idx = CLASS_TO_IDX
    dataset.samples = [
        (p, CLASS_TO_IDX[Path(p).parent.name])
        for p, _ in dataset.samples
    ]
    dataset.targets = [label for _, label in dataset.samples]
    return dataset


def get_datasets(train_dir, val_dir, image_size=224):
    """Build datasets for train and val. Folders must be named defect and no_defect."""
    train_transforms = get_transforms(image_size, is_training=True)
    val_transforms = get_transforms(image_size, is_training=False)

    train_ds = _apply_class_mapping(datasets.ImageFolder(train_dir, transform=train_transforms))
    val_ds = _apply_class_mapping(datasets.ImageFolder(val_dir, transform=val_transforms))

    return train_ds, val_ds


def get_dataloaders(train_dir, val_dir, batch_size=32, image_size=224, num_workers=4):
    """Return train and val DataLoaders from separate train and val folders."""
    train_ds, val_ds = get_datasets(train_dir, val_dir, image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, train_ds.class_to_idx


class PathLabelDataset(Dataset):
    """Dataset from a list of (path, label). Used for MVTec-style layout."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def _collect_images(folder):
    paths = []
    folder = Path(folder)
    if not folder.exists():
        return paths
    for ext in IMAGE_EXTENSIONS:
        paths.extend(folder.glob(f"*{ext}"))
    return sorted(paths)


def get_mvtec_dataloaders(
    category_dir,
    batch_size=32,
    image_size=224,
    num_workers=4,
    train_ratio=0.8,
    seed=42,
):
    """
    Use MVTec AD as-is: category_dir has train/ (good) and test/ (good + defect types).
    No copying. Train/val split is done in code.
    """
    category_dir = Path(category_dir)
    train_dir = category_dir / "train"
    test_dir = category_dir / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"MVTec category folder must contain train/ and test/. Check: {category_dir}"
        )

    # Good = train/*.png, train/good/*.png, and test/good/*.png
    good_paths = _collect_images(train_dir)
    train_good = train_dir / "good"
    if train_good.exists():
        good_paths.extend(_collect_images(train_good))
    good_test = test_dir / "good"
    if good_test.exists():
        good_paths.extend(_collect_images(good_test))

    # Defect = all test subfolders except "good"
    defect_paths = []
    for sub in test_dir.iterdir():
        if sub.is_dir() and sub.name != "good":
            defect_paths.extend(_collect_images(sub))

    if not good_paths or not defect_paths:
        raise ValueError(
            f"Need both good and defect images. Found good: {len(good_paths)}, defect: {len(defect_paths)} in {category_dir}"
        )

    class_to_idx = CLASS_TO_IDX
    samples = [(p, 0) for p in good_paths] + [(p, 1) for p in defect_paths]
    random.Random(seed).shuffle(samples)

    n = len(samples)
    split = int(n * train_ratio)
    train_samples = samples[:split]
    val_samples = samples[split:]

    train_transforms = get_transforms(image_size, is_training=True)
    val_transforms = get_transforms(image_size, is_training=False)
    train_ds = PathLabelDataset(train_samples, transform=train_transforms)
    val_ds = PathLabelDataset(val_samples, transform=val_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, class_to_idx


def check_mvtec_category(category_dir):
    """Check that category_dir has train/ and test/ with images."""
    category_dir = Path(category_dir)
    if not category_dir.exists():
        return False, f"Directory not found: {category_dir}"
    if not (category_dir / "train").exists():
        return False, "Missing train/ folder"
    if not (category_dir / "test").exists():
        return False, "Missing test/ folder"
    train_dir = category_dir / "train"
    good = _collect_images(train_dir)
    if (train_dir / "good").exists():
        good.extend(_collect_images(train_dir / "good"))
    defect = []
    for sub in (category_dir / "test").iterdir():
        if sub.is_dir() and sub.name != "good":
            defect.extend(_collect_images(sub))
    if not good:
        return False, "No images in train/"
    if not defect:
        return False, "No defect images in test/ (only test/good found)"
    return True, {"good": len(good), "defect": len(defect)}


def get_mvtec_categories(mvtec_root):
    """List category names that have train/ and test/ (for --all-categories)."""
    root = Path(mvtec_root)
    if not root.exists():
        return []
    categories = []
    for path in root.iterdir():
        if path.is_dir() and (path / "train").exists() and (path / "test").exists():
            ok, _ = check_mvtec_category(path)
            if ok:
                categories.append(path.name)
    return sorted(categories)


def check_data_structure(root_dir):
    """Check that defect and no_defect folders exist."""
    root = Path(root_dir)
    if not root.exists():
        return False, f"Directory not found: {root_dir}"
    subdirs = [d.name for d in root.iterdir() if d.is_dir()]
    required = {"defect", "no_defect"}
    missing = required - set(subdirs)
    if missing:
        return False, f"Missing folders: {missing}. Expected: defect, no_defect"
    return True, subdirs
