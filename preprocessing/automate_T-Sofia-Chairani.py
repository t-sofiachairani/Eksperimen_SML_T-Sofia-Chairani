import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def list_classes(split_dir: Path) -> list[str]:
    classes = [d.name for d in split_dir.iterdir() if d.is_dir()]
    classes.sort()
    return classes


def load_split(split_dir: Path, classes: list[str], img_size: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for idx, cls in enumerate(classes):
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            continue
        for img_path in cls_dir.glob("*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True, help="Folder berisi train/validation/test")
    parser.add_argument("--out_dir", type=str, required=True, help="Output folder preprocessing")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    train_dir = raw_dir / "train"
    val_dir = raw_dir / "validation"
    test_dir = raw_dir / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"train folder tidak ditemukan: {train_dir}")

    classes = list_classes(train_dir)
    (out_dir).mkdir(parents=True, exist_ok=True)
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes}, f, ensure_ascii=False, indent=2)

    for split_name, split_dir in [("train", train_dir), ("validation", val_dir), ("test", test_dir)]:
        if not split_dir.exists():
            print(f"[SKIP] split '{split_name}' tidak ada di {split_dir}")
            continue

        print(f"Processing split: {split_name}")
        X, y = load_split(split_dir, classes, args.img_size)
        save_npz(out_dir, split_name, X, y)
        print(f"Saved {split_name}: X={X.shape}, y={y.shape}")

    print("Done. Output:", out_dir.resolve())


if __name__ == "__main__":
    main()
