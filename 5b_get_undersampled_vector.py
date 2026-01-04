#!/usr/bin/env python3
import argparse
from pathlib import Path
import pickle
import re


def iter_images_flat(p: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    for f in p.iterdir():
        if f.is_file() and f.suffix.lower() in exts:
            yield f


def extract_index_from_filename(p: Path):
    """
    Extract first integer from filename stem.
    Example: img_00123.jpg -> 123
    """
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--undersampling_dir", default="undersampling", type=str,
                    help="Root undersampling directory containing images/")
    ap.add_argument("--concepts_in", default="concept_vectors.pkl", type=str,
                    help="Original concept_vectors.pkl")
    ap.add_argument("--out_pkl", default="undersampled_image_vector_pairs.pkl", type=str,
                    help="Output pickle: list of (image_name, vector)")
    ap.add_argument("--sort", action="store_true",
                    help="Sort image names for deterministic ordering")
    ap.add_argument("--strict", action="store_true",
                    help="Fail if any undersampled image has no concept vector")
    args = ap.parse_args()

    root = Path(args.undersampling_dir)
    images_dir = root / "images"
    if not images_dir.exists():
        raise SystemExit(f"Missing directory: {images_dir}")

    images = [p for p in iter_images_flat(images_dir)]
    if not images:
        raise SystemExit(f"No images found in {images_dir}")

    if args.sort:
        images.sort(key=lambda p: p.name)

    # Load concepts
    with Path(args.concepts_in).open("rb") as f:
        concepts = pickle.load(f)

    pairs = []
    missing = []

    # --- Case 1: concepts is a list (index-aligned) ---
    if isinstance(concepts, list):
        n = len(concepts)
        for p in images:
            idx = extract_index_from_filename(p)
            if idx is None or idx < 0 or idx >= n:
                missing.append(p.name)
                continue
            pairs.append((p.name, concepts[idx]))

    # --- Case 2: concepts is a dict keyed by filename or index ---
    elif isinstance(concepts, dict):
        # detect key type by sampling
        any_key = next(iter(concepts.keys()), None)
        if any_key is None:
            raise SystemExit("concept_vectors.pkl dict is empty")

        # Filename-keyed dict
        if isinstance(any_key, str):
            for p in images:
                vec = concepts.get(p.name)
                if vec is None:
                    missing.append(p.name)
                    continue
                pairs.append((p.name, vec))

        # Index-keyed dict
        elif isinstance(any_key, int):
            for p in images:
                idx = extract_index_from_filename(p)
                if idx is None:
                    missing.append(p.name)
                    continue
                vec = concepts.get(idx)
                if vec is None:
                    missing.append(p.name)
                    continue
                pairs.append((p.name, vec))
        else:
            raise TypeError(f"Unsupported dict key type in concepts: {type(any_key)}")

    else:
        raise TypeError(f"Unsupported concept_vectors.pkl type: {type(concepts)}")

    print(f"Images in undersampling/images: {len(images)}")
    print(f"Pairs written: {len(pairs)}")
    print(f"Missing concept vectors: {len(missing)}")

    if missing[:10]:
        print("Examples missing:", missing[:10])

    if args.strict and missing:
        raise SystemExit("Strict mode: some undersampled images had no concept vector.")

    # Save pairs
    out_path = Path(args.out_pkl)
    with out_path.open("wb") as f:
        pickle.dump(pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

    data = pickle.load(open('undersampled_image_vector_pairs.pkl', 'rb'))
    print(len(data))