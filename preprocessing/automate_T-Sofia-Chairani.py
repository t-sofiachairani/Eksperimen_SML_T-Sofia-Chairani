import argparse
import glob
import json
from pathlib import Path

import pandas as pd


KNOWN_CLASSES = {
    "kue dadar gulung",
    "kue kastengel",
    "kue klepon",
    "kue lapis",
    "kue lumpur",
    "kue putri salju",
    "kue risoles",
    "kue serabi",
}


def normalize_label(s: str) -> str:
    return s.strip().lower().replace("_", " ").replace("-", " ")


def find_dataset_root(raw_dir: Path) -> Path:
    """Return folder yang punya train/validation/test.
    Kalau raw_dir belum punya train, coba cari 1 level di bawahnya.
    """
    if (
        (raw_dir / "train").exists()
        and (raw_dir / "validation").exists()
        and (raw_dir / "test").exists()
    ):
        return raw_dir

    # cari satu level di bawah (misalnya raw_dir/kue_raw/...)
    for child in raw_dir.iterdir():
        if child.is_dir():
            if (
                (child / "train").exists()
                and (child / "validation").exists()
                and (child / "test").exists()
            ):
                return child

    raise FileNotFoundError(
        f"Tidak menemukan folder train/validation/test di: {raw_dir} (atau 1 level di bawahnya)."
    )


def collect_split(split_dir: Path, split_name: str) -> pd.DataFrame:
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        image_paths += glob.glob(str(split_dir / "**" / ext), recursive=True)

    if not image_paths:
        raise RuntimeError(f"Tidak ada gambar ditemukan di: {split_dir}")

    rows = []
    for p in image_paths:
        pth = Path(p)
        label = normalize_label(pth.parent.name)
        rows.append({"filepath": str(pth), "label": label, "split": split_name})

    df = pd.DataFrame(rows)
    df = df[df["label"].isin(KNOWN_CLASSES)].reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Folder dataset (berisi train/validation/test)",
    )
    ap.add_argument(
        "--out_dir", type=str, default="output", help="Folder output (default: output)"
    )
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_root = find_dataset_root(raw_dir)
    train_dir = ds_root / "train"
    val_dir = ds_root / "validation"
    test_dir = ds_root / "test"

    train_df = collect_split(train_dir, "train")
    val_df = collect_split(val_dir, "val")
    test_df = collect_split(test_dir, "test")

    df_all = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # label mapping (harus dibuat sebelum export JSON)
    class_names = sorted(df_all["label"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    # export CSV
    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)
    test_df.to_csv(out_dir / "test_split.csv", index=False)

    # export JSON mapping
    with open(out_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(
            {"class_to_idx": class_to_idx, "idx_to_class": idx_to_class},
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"classes": class_names}, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("Dataset root :", ds_root.resolve())
    print("Output folder:", out_dir.resolve())
    print("Files created:")
    for p in sorted(out_dir.glob("*")):
        print("-", p.name)


if __name__ == "__main__":
    main()
