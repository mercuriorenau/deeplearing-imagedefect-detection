# Recommended datasets for Image Defect Detection

---

## 1. MVTec AD (best for industry / portfolios)

- **What:** Industrial defect detection. 15 object categories (bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper). Each category has good images and several defect types.
- **Size:** ~5 GB total.
- **Download:** [MVTec AD downloads](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads) (free, registration).
- **How to use:** Extract categories under `data/` as-is (see [data/README.md](../data/README.md)). No copying into `defect/` / `no_defect/` folders.

```bash
python -m src.train --all-categories
```

- **Why:** Standard benchmark, real industrial images, matches this project's multi-category workflow.

---

## 2. Jar Lids (product defects) — quick custom test

- **What:** Jar lids: damaged vs intact. ~900 damaged, ~960 intact.
- **Where:** Kaggle — [Jar Lids / product defects](https://www.kaggle.com/datasets/rrighart/jarlids).
- **How to use:** Map intact → `data/train/no_defect/` and `data/val/no_defect/`; damaged → `data/train/defect/` and `data/val/defect/`. Split train/val (e.g. 80/20 per class).
- **Why:** Small, binary, good for testing the custom-folder path quickly.

---

## 3. Kaggle — Product Defect Detection (competition)

- **What:** Product images, binary target: defective vs non-defective.
- **Where:** [Product Defect Detection | Kaggle](https://www.kaggle.com/competitions/product-defect-detection).
- **How to use:** Put label-0 images in `no_defect`, label-1 in `defect`. Split into `data/train/` and `data/val/`.
- **Why:** Directly defect vs no defect for the custom-folder layout.

---

## 4. Wood surface defects (texture)

- **What:** Large set of wood surface defect images.
- **Where:** Kaggle — search **wood surface defects**.
- **How to use:** Map classes to `defect` and `no_defect`, then use the custom-folder layout under `data/train/` and `data/val/`.

---

## Quick choice

| Goal | Dataset to use |
|------|----------------|
| Industry standard, this repo's default | **MVTec AD** (as-is under `data/`) |
| Fast test, few images | **Jar Lids** (custom folders) |
| Binary competition style | **Product Defect Detection** (Kaggle) |
| Texture / surface | **Wood surface defects** |

**MVTec AD:** use native layout — see [data/README.md](../data/README.md).

**Custom datasets:** use this layout:

```
data/train/defect/
data/train/no_defect/
data/val/defect/
data/val/no_defect/
```

Then run: `python -m src.train`
