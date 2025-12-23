import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def list_classes(train_dir: Path) -> list[str]:
    classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    classes.sort()
    return classes


def iter_images(cls_dir: Path):
    for p in cls_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def load_split_to_npz(split_dir: Path, classes: list[str], img_size: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for idx, cls in enumerate(classes):
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            continue
        for img_path in iter_images(cls_dir):
            img = Image.open(img_path).convert("RGB")
            img = img.resize((img_size, img_size))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            X.append(arr)
            y.append(idx)

    X = np.stack(X) if X else np.empty((0, img_size, img_size, 3), dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y


def save_npz(out_dir: Path, split: str, X: np.ndarray, y: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{split}.npz", X=X, y=y)


def build_split_csv(split_dir: Path, split_name: str) -> pd.DataFrame:
    rows = []
    # split_dir berisi folder kelas
    for cls_dir in split_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        label = cls_dir.name
        for img_path in iter_images(cls_dir):
            rows.append(
                {
                    "filepath": str(img_path),
                    "label": label,
                    "split": split_name,
                }
            )
    return pd.DataFrame(rows)


def find_dataset_root(raw_dir: Path) -> Path:
    """
    Mencari folder yang punya train/validation/test.
    - Kalau raw_dir langsung punya train -> pakai raw_dir
    - Kalau tidak, cari 1 level di bawahnya
    """
    if (raw_dir / "train").exists():
        return raw_dir

    for child in raw_dir.iterdir():
        if child.is_dir() and (child / "train").exists():
            return child

    raise FileNotFoundError(
        f"Tidak menemukan folder 'train' di: {raw_dir} (atau 1 level di bawahnya)."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True, help="Folder dataset (punya train/validation/test)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output folder preprocessing (mis. output)")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    raw_dir = find_dataset_root(Path(args.raw_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dir = raw_dir / "train"
    val_dir = raw_dir / "validation"
    test_dir = raw_dir / "test"

    classes = list_classes(train_dir)

    # mapping
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    (out_dir / "label_map.json").write_text(
        json.dumps({"classes": classes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "class_mapping.json").write_text(
        json.dumps({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # CSV split
    train_df = build_split_csv(train_dir, "train")
    val_df = build_split_csv(val_dir, "val") if val_dir.exists() else pd.DataFrame(columns=["filepath", "label", "split"])
    test_df = build_split_csv(test_dir, "test") if test_dir.exists() else pd.DataFrame(columns=["filepath", "label", "split"])

    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)
    test_df.to_csv(out_dir / "test_split.csv", index=False)

    # NPZ
    for split_name, split_dir in [("train", train_dir), ("validation", val_dir), ("test", test_dir)]:
        if not split_dir.exists():
            print(f"[SKIP] split '{split_name}' tidak ada: {split_dir}")
            continue

        print(f"Processing NPZ split: {split_name}")
        X, y = load_split_to_npz(split_dir, classes, args.img_size)
        save_npz(out_dir, split_name, X, y)
        print(f"Saved {split_name}.npz: X={X.shape}, y={y.shape}")

    print("\nDONE. Output folder:", out_dir.resolve())
    print("Files expected:")
    print("- train_split.csv, val_split.csv, test_split.csv")
    print("- label_map.json, class_mapping.json")
    print("- train.npz, validation.npz, test.npz")


if __name__ == "__main__":
    main()
