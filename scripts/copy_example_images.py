"""
Copy one good and one defect image per category from data/ to examples/.
These are the same images the Streamlit app uses as reference. Run once so
you can push only examples/ to git (not the full dataset).
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EXAMPLES = ROOT / "examples"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def collect_first(path):
    for ext in IMAGE_EXTENSIONS:
        files = sorted(path.glob(f"*{ext}"))
        if files:
            return files[0]
    return None


def main():
    if not DATA.exists():
        print("No data/ folder. Put MVTec categories in data/ first.")
        return
    for cat_dir in sorted(DATA.iterdir()):
        if not cat_dir.is_dir():
            continue
        train_good = cat_dir / "train" / "good"
        if not train_good.exists():
            train_good = cat_dir / "test" / "good"
        if not train_good.exists():
            continue
        good_src = collect_first(train_good)
        defect_src = None
        test_dir = cat_dir / "test"
        if test_dir.exists():
            for sub in sorted(test_dir.iterdir()):
                if sub.is_dir() and sub.name != "good":
                    defect_src = collect_first(sub)
                    break
        if not good_src or not defect_src:
            continue
        out_dir = EXAMPLES / cat_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(good_src, out_dir / "good.png")
        shutil.copy(defect_src, out_dir / "defect.png")
        print(f"  {cat_dir.name}: good.png, defect.png")
    print("Done. examples/ ready to commit (used by app as reference images).")


if __name__ == "__main__":
    main()
