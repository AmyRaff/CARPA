#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path
from collections import Counter, defaultdict


def safe_link_or_copy(src: Path, dst: Path, prefer_hardlink: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if prefer_hardlink:
        try:
            os.link(src, dst)  # fast, no extra disk usage if same filesystem
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="undersampling/images", type=str,
                    help="Directory containing the undersampled images")
    ap.add_argument("--split_file", default="train_test_split.txt", type=str,
                    help="File with lines: 'im_name label data_split'")
    ap.add_argument("--out_root", default=".", type=str,
                    help="Where to create train/ val/ test/ directories")
    ap.add_argument("--hardlink", action="store_true",
                    help="Prefer hardlinks (falls back to copy)")
    ap.add_argument("--clean", action="store_true",
                    help="Delete existing train/ val/ test directories before writing")
    ap.add_argument("--skip_missing", action="store_true",
                    help="Skip entries whose image file isn't found in images_dir (otherwise error)")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    split_file = Path(args.split_file)
    out_root = Path(args.out_root)

    if not images_dir.exists():
        raise SystemExit(f"Missing images_dir: {images_dir}")
    if not split_file.exists():
        raise SystemExit(f"Missing split_file: {split_file}")

    # Optional clean
    if args.clean:
        for split in ("train", "val", "test"):
            d = out_root / split
            if d.exists():
                shutil.rmtree(d)

    # Build fast lookup for source images by filename
    # (flat directory expected)
    src_map = {p.name: p for p in images_dir.iterdir() if p.is_file()}

    created = Counter()
    per_split_label = defaultdict(Counter)
    missing = []
    bad_lines = 0

    with split_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                bad_lines += 1
                continue

            im_name, label, split = parts[0], parts[1], parts[2].lower()
            if split not in {"train", "val", "test"}:
                bad_lines += 1
                continue

            src = src_map.get(im_name)
            if src is None:
                missing.append(im_name)
                if args.skip_missing:
                    continue
                raise SystemExit(f"Image '{im_name}' (line {line_no}) not found in {images_dir}")

            dst = out_root / split / label / im_name
            mode = safe_link_or_copy(src, dst, args.hardlink)

            created[split] += 1
            per_split_label[split][label] += 1
            created[f"{split}:{mode}"] += 1

    print("Done.")
    print("Created counts:", dict(created))
    if bad_lines:
        print(f"[WARN] Skipped {bad_lines} malformed/invalid lines in split file.")
    if missing:
        print(f"[WARN] Missing images referenced by split file: {len(missing)} (e.g. {missing[:10]})")

    print("\nPer-split label counts:")
    for split in ("train", "val", "test"):
        if per_split_label[split]:
            print(f"  {split}: {dict(per_split_label[split])}")


if __name__ == "__main__":
    main()
