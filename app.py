"""
Web UI: upload images, choose what we're detecting (category), get defect / no defect.
Run: streamlit run app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st
from PIL import Image

from src.predict import load_model, get_inference_transform, predict_image_from_pil


def get_available_models():
    """List checkpoints: best_model_<category>.pt only (skip generic best_model.pt)."""
    ckpt_dir = ROOT / "checkpoints"
    if not ckpt_dir.exists():
        return []
    models = []
    for p in ckpt_dir.glob("best_model*.pt"):
        if p.stem == "best_model":
            continue  # skip generic checkpoint so dropdown only shows categories
        name = p.stem.replace("best_model_", "")
        models.append((name, str(p)))
    return sorted(models, key=lambda x: x[0])


# Short description per category for instructions
CATEGORY_HINT = {
    "bottle": "Bottles (e.g. glass or plastic). Upload a clear photo of the product.",
    "cable": "Cables and wires. Good or with bends, cuts, or other defects.",
    "capsule": "Capsules (e.g. pills). Intact or damaged.",
    "carpet": "Carpet or fabric surfaces. Clean or with cuts, stains, holes.",
    "grid": "Grid patterns. Regular or with defects.",
    "hazelnut": "Hazelnuts. Whole or with cracks, holes.",
    "leather": "Leather surfaces. Good or with scratches, cuts.",
    "metal_nut": "Metal nuts. Correct or with defects.",
    "pill": "Pills. Intact or defective.",
    "screw": "Screws. Good or damaged.",
    "tile": "Tiles. Clean or with cracks, glues, contamination.",
    "toothbrush": "Toothbrushes. New or defective.",
    "transistor": "Transistors. Good or with anomalies.",
    "wood": "Wood surfaces. Clean or with defects.",
    "zipper": "Zippers. Working or with defects.",
}


def get_reference_images(category):
    """Return (good_example_path, defect_example_path). Prefer examples/<category>/ then data/<category>/."""
    # Prefer bundled examples (so clone works without full dataset)
    ex_dir = ROOT / "examples" / category
    if ex_dir.exists():
        good_p = ex_dir / "good.png"
        defect_p = ex_dir / "defect.png"
        if good_p.exists() and defect_p.exists():
            return good_p, defect_p
    # Fallback: full dataset under data/
    data_dir = ROOT / "data" / category
    if not data_dir.exists():
        return None, None
    good_dir = data_dir / "train" / "good"
    if not good_dir.exists():
        good_dir = data_dir / "test" / "good"
    good_path = None
    if good_dir.exists():
        for ext in (".png", ".jpg", ".jpeg"):
            files = list(good_dir.glob(f"*{ext}"))
            if files:
                good_path = files[0]
                break
    defect_path = None
    test_dir = data_dir / "test"
    if test_dir.exists():
        for sub in test_dir.iterdir():
            if sub.is_dir() and sub.name != "good":
                for ext in (".png", ".jpg", ".jpeg"):
                    files = list(sub.glob(f"*{ext}"))
                    if files:
                        defect_path = files[0]
                        break
                if defect_path:
                    break
    return good_path, defect_path


def main():
    st.set_page_config(page_title="Image Defect Detection", layout="centered")
    st.title("Image Defect Detection")
    st.markdown("Upload a photo, choose **what we're detecting**, and see if it has a defect or not.")

    with st.expander("How to use", expanded=True):
        st.markdown("""
        1. **Choose the object type** in the dropdown (bottle, cable, carpet, etc.). Each option has a model trained for that product.
        2. **Upload one or more photos** of that object (good or with defects).
        3. The model will say **Defect** or **No defect** and a confidence percentage.
        """)

    models = get_available_models()
    if not models:
        st.warning(
            "No trained models found. Train first:\n\n"
            "`python -m src.train --all-categories`\n\n"
            "Or one category: `python -m src.train --category bottle`"
        )
        return

    # What we're detecting = which model (category)
    options = [m[0] for m in models]
    checkpoint_paths = {m[0]: m[1] for m in models}
    selected = st.selectbox(
        "What are we detecting?",
        options=options,
        help="Choose the type of object. Each option has a model trained for that product.",
    )
    checkpoint_path = checkpoint_paths[selected]

    # Load model once and cache
    @st.cache_resource
    def load_cached_model(path):
        device = __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")
        model, idx_to_class, image_size = load_model(path, device=device)
        transform = get_inference_transform(image_size)
        return model, idx_to_class, transform, device

    model, idx_to_class, transform, device = load_cached_model(checkpoint_path)

    # Upload
    uploaded = st.file_uploader("Upload image(s)", type=["png", "jpg", "jpeg", "bmp", "webp"], accept_multiple_files=True)

    # Instructions and reference for this category
    hint = CATEGORY_HINT.get(selected, "Upload a clear photo of the product.")
    st.caption(hint)
    good_ref, defect_ref = get_reference_images(selected)
    if good_ref or defect_ref:
        ref_cols = st.columns(2)
        if good_ref:
            with ref_cols[0]:
                try:
                    st.image(str(good_ref), caption="Example: no defect", use_container_width=True)
                except OSError as exc:
                    st.warning(f"Could not load good example: {exc}")
        if defect_ref:
            with ref_cols[1]:
                try:
                    st.image(str(defect_ref), caption="Example: defect", use_container_width=True)
                except OSError as exc:
                    st.warning(f"Could not load defect example: {exc}")
    st.divider()

    if not uploaded:
        st.info("Upload one or more images to analyze.")
        return

    st.subheader("Results")
    for f in uploaded:
        img = Image.open(f).convert("RGB")
        label, conf = predict_image_from_pil(model, img, transform, device, idx_to_class)
        is_defect = label == "defect"
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img, use_container_width=True)
        with col2:
            st.markdown(f"**Object type:** {selected}")
            if is_defect:
                st.error(f"**Result: Defect** (confidence {conf:.0%})")
            else:
                st.success(f"**Result: No defect** (confidence {conf:.0%})")
        st.divider()


if __name__ == "__main__":
    main()
