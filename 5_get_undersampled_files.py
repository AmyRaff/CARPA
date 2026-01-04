#!/usr/bin/env python3
import argparse
from pathlib import Path
from collections import defaultdict


def iter_images_flat(p: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    for f in p.iterdir():
        if f.is_file() and f.suffix.lower() in exts:
            yield f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--undersampling_dir", default="undersampling", type=str,
                    help="Root dir containing images/ and per-label subdirs")
    ap.add_argument("--out_file", default="image_class_labels.txt", type=str,
                    help="Output labels file (image_name label)")
    ap.add_argument("--strict", action="store_true",
                    help="Fail if any image is missing a label or has multiple labels")
    ap.add_argument("--sort", action="store_true",
                    help="Sort output by image name (default: keep scanning order)")
    args = ap.parse_args()

    root = Path(args.undersampling_dir)
    images_dir = root / "images"
    if not images_dir.exists():
        raise SystemExit(f"Missing directory: {images_dir}")

    # Collect all kept image names
    kept_names = {p.name for p in iter_images_flat(images_dir)}
    if not kept_names:
        raise SystemExit(f"No images found in {images_dir}")

    # Build mapping from image name -> list of labels (to detect conflicts)
    name_to_labels = defaultdict(list)

    for d in root.iterdir():
        if not d.is_dir():
            continue
        if d.name == "images":
            continue
        label = d.name
        for p in iter_images_flat(d):
            # Only consider images that are in undersampling/images
            if p.name in kept_names:
                name_to_labels[p.name].append(label)

    # Validate + build final mapping
    missing = []
    conflicts = []
    final = {}

    for name in kept_names:
        labs = name_to_labels.get(name, [])
        if len(labs) == 0:
            missing.append(name)
        elif len(labs) > 1:
            conflicts.append((name, labs))
        else:
            final[name] = labs[0]

    # Report
    print(f"Images in {images_dir}: {len(kept_names)}")
    print(f"Labeled (unique): {len(final)}")
    print(f"Missing label: {len(missing)}")
    print(f"Conflicts (multi-label): {len(conflicts)}")

    if missing[:10]:
        print("Examples missing:", missing[:10])
    if conflicts[:5]:
        print("Examples conflicts:", conflicts[:5])

    if args.strict and (missing or conflicts):
        raise SystemExit("Strict mode: found missing labels and/or conflicts.")

    # Write output
    out_path = Path(args.out_file)
    items = list(final.items())
    if args.sort:
        items.sort(key=lambda x: x[0])

    with out_path.open("w", encoding="utf-8") as f:
        for name, label in items:
            f.write(f"{name} {label}\n")

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
