"""
prepare_dataset.py — normalize the raw Roboflow/Kaggle glove dataset into a small,
canonical YOLO-format training set with our two required class names.

Source dataset: rsaishivani/hand-glove-dataset (Kaggle, MIT license), a mirror of the
Roboflow project glove-uylxg/glove-q7czq. It already ships in YOLOv8 txt format with
classes ['glove', 'no_glove'] — a clean 1:1 rename to ['gloved_hand', 'bare_hand'],
no remapping/dropping logic is needed. Class balance in the source train split is
already near-even (792 / 765 boxes), so a plain random subsample is safe — no
stratified sampling required.

Usage:
    python prepare_dataset.py --src glove_raw --dst data --max-train 400 --max-val 100
"""
import argparse
import random
import shutil
from pathlib import Path

# Canonical class contract — index 0/1 frozen across Part 1 and Part 2.
CANONICAL = ["gloved_hand", "bare_hand"]
# Source class index -> canonical index. The source dataset's classes.txt is
# ['glove', 'no_glove'], which already lines up 1:1 with our canonical order.
SOURCE_CLASSES = ["glove", "no_glove"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="Root of the extracted raw dataset (has train/valid/test)")
    p.add_argument("--dst", default="data", help="Output directory for the canonical dataset")
    p.add_argument("--max-train", type=int, default=400, help="Max number of training images to keep")
    p.add_argument("--max-val", type=int, default=100, help="Max number of validation images to keep")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def collect_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    """Return (image_path, label_path) pairs that both exist for a split."""
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    pairs = []
    for img in sorted(img_dir.glob("*.*")):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def class_distribution(pairs: list[tuple[Path, Path]]) -> dict[int, int]:
    dist: dict[int, int] = {}
    for _, lbl in pairs:
        for line in lbl.read_text().strip().splitlines():
            if not line.strip():
                continue
            c = int(line.split()[0])
            dist[c] = dist.get(c, 0) + 1
    return dist


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    src = Path(args.src)
    dst = Path(args.dst)

    print(f"Canonical classes (frozen contract): {CANONICAL}")
    print(f"Source classes (already aligned 1:1): {SOURCE_CLASSES}")
    print()

    train_pairs = collect_pairs(src / "train")
    val_pairs = collect_pairs(src / "valid")

    print(f"Source counts: train={len(train_pairs)}  valid={len(val_pairs)}")
    print(f"Source train class distribution (box counts): {class_distribution(train_pairs)}")
    print(f"Source valid class distribution (box counts): {class_distribution(val_pairs)}")
    print()

    random.shuffle(train_pairs)
    random.shuffle(val_pairs)
    train_pairs = train_pairs[: args.max_train]
    val_pairs = val_pairs[: args.max_val]

    # Layout: data/images/{train,val}, data/labels/{train,val}
    for split_name, pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_out = dst / "images" / split_name
        lbl_out = dst / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img, lbl in pairs:
            shutil.copy2(img, img_out / img.name)
            shutil.copy2(lbl, lbl_out / lbl.name)

    data_yaml = dst / "data.yaml"
    data_yaml.write_text(
        f"path: {dst.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: {CANONICAL[0]}\n"
        f"  1: {CANONICAL[1]}\n"
    )

    print(f"Subset counts: train={len(train_pairs)}  val={len(val_pairs)}")
    print(f"Subset train class distribution (box counts): {class_distribution(train_pairs)}")
    print(f"Subset val class distribution (box counts): {class_distribution(val_pairs)}")
    print()
    print(f"Wrote {data_yaml}")


if __name__ == "__main__":
    main()
