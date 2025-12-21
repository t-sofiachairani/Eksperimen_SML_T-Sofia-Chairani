import os
import re
import sys
import zipfile
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

def safe_extract(zip_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    counters = {}  # key: (split, cls) -> idx

    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            p = Path(name)

            # skip folder entries
            if name.endswith("/"):
                continue

            ext = p.suffix.lower()
            if ext not in IMG_EXT:
                continue

            parts = p.parts

            # cari posisi split (train/test/validation) di path zip (kadang ada prefix folder)
            split_idx = None
            for i, part in enumerate(parts):
                if part in ("train", "test", "validation"):
                    split_idx = i
                    break
            if split_idx is None:
                continue

            split = parts[split_idx]
            if len(parts) <= split_idx + 1:
                continue
            cls = parts[split_idx + 1]

            key = (split, cls)
            counters[key] = counters.get(key, 0) + 1
            idx = counters[key]

            # bikin nama file pendek: 000001.jpg
            dest_dir = out_dir / split / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / f"{idx:06d}{ext}"

            # stream extract
            with z.open(name) as src, open(dest_path, "wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)

    print("DONE. Extracted to:", out_dir.resolve())

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_kue_shortnames.py <zip_path> <out_dir>")
        sys.exit(1)

    zip_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])

    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    safe_extract(zip_path, out_dir)
