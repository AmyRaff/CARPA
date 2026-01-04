#!/usr/bin/env python3
import argparse
import pickle
import random
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Set, Any, Tuple, Optional


# -------------------------
# Your label mapping (as provided)
# -------------------------
def get_label_from_vector(vector: Dict[str, int]) -> List[str]:
    labels = []
    if vector.get('unremarkable', 0) == 1:
        return ['NoRelevantFinding']
    if vector.get('mass', 0) == 1 or vector.get('nodule', 0) == 1 or vector.get('irregular_hilum', 0) == 1 \
        or vector.get('adenopathy', 0) == 1 or vector.get('irregular_parenchyma', 0) == 1:
            labels.append('SuspiciousMalignancy')
    if vector.get('pneumonitis', 0) == 1 or vector.get('consolidation', 0) == 1 or vector.get('infection', 0) == 1:
        labels.append('Pneumonia')
    # exclude opacities - is dependant on others anyway
    if vector.get('effusion', 0) == 1 or vector.get('fluid', 0) == 1 or vector.get('meniscus_sign', 0) == 1 \
        or vector.get('costophrenic_angle', 0) == 1:
            labels.append('Effusion')
    if vector.get('absent_lung_markings', 0) == 1 or vector.get('pleural_air', 0) == 1:
        labels.append('Pneumothorax')
    if vector.get('irregular_diaphragm', 0) == 1 and vector.get('lung_collapse', 0) == 1:
        labels.append('Pneumothorax')
    if vector.get('enlarged_heart', 0) == 1:
        labels.append('Cardiomegaly')

    return list(set(labels))


# -------------------------
# Concept groups per label
# -------------------------
LABEL_TO_CONCEPTS: Dict[str, List[str]] = {
    "NoRelevantFinding": ["unremarkable"],
    "SuspiciousMalignancy": ["mass", "nodule", "irregular_hilum", "adenopathy", "irregular_parenchyma"],
    "Pneumonia": ["pneumonitis", "consolidation", "infection"],
    "Effusion": ["effusion", "fluid", "meniscus_sign", "costophrenic_angle"],
    "Pneumothorax": ["absent_lung_markings", "pleural_air", "irregular_diaphragm", "lung_collapse"],
    "Cardiomegaly": ["enlarged_heart"],
}
ALL_LABELS = list(LABEL_TO_CONCEPTS.keys())


# -------------------------
# Utilities
# -------------------------
def ensure_all_keys_present(vec: Dict[str, Any]) -> Dict[str, int]:
    out = dict(vec)
    for keys in LABEL_TO_CONCEPTS.values():
        for k in keys:
            out[k] = int(out.get(k, 0))
    return out


def set_concepts(vec: Dict[str, int], keys: List[str], value: int) -> None:
    v = int(value)
    for k in keys:
        vec[k] = v


def random_set_some_to_one(vec: Dict[str, int], keys: List[str], k_min: int, k_max: int, rng: random.Random) -> None:
    if not keys:
        return
    k_max = min(k_max, len(keys))
    k_min = max(1, min(k_min, k_max))
    k = rng.randint(k_min, k_max)
    for c in rng.sample(keys, k):
        vec[c] = 1


def clear_all_non_unremarkable(vec: Dict[str, int]) -> None:
    for label, keys in LABEL_TO_CONCEPTS.items():
        if label == "NoRelevantFinding":
            continue
        set_concepts(vec, keys, 0)


def vec_signature(vec: Dict[str, int]) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted((k, int(v)) for k, v in vec.items()))


def active_concepts(vec: Dict[str, int]) -> List[str]:
    return sorted([k for k, v in vec.items() if int(v) == 1])


def pretty_labels(vec: Dict[str, int]) -> List[str]:
    return sorted(get_label_from_vector(vec))


# -------------------------
# I/O
# -------------------------
def load_test_filenames(split_file: Path) -> Set[str]:
    test_names: Set[str] = set()
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            img, _, split = parts[0], parts[1], parts[2].lower()
            if split == "test":
                test_names.add(img)
    if not test_names:
        raise RuntimeError(f"No test entries found in {split_file}")
    return test_names


def load_concept_map(concepts_pkl: Path) -> Dict[str, Dict[str, int]]:
    """
    Supports tuple structure: (img_name, (something, concept_dict))
    Also tolerates: (img_name, concept_dict)
    """
    with concepts_pkl.open("rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, list):
        raise TypeError(f"Expected list in {concepts_pkl}, got {type(obj)}")

    out: Dict[str, Dict[str, int]] = {}
    for item in obj:
        if not (isinstance(item, tuple) and len(item) == 2):
            continue
        img_name, payload = item
        if not isinstance(img_name, str):
            continue

        vec = None
        if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[1], dict):
            vec = payload[1]
        elif isinstance(payload, dict):
            vec = payload

        if isinstance(vec, dict):
            out[img_name] = ensure_all_keys_present(vec)

    if not out:
        raise RuntimeError(f"Could not parse any concept vectors from {concepts_pkl}")
    return out


def pick_other_label(current: Set[str], rng: random.Random) -> str:
    choices = [l for l in ALL_LABELS if l not in current and l != "NoRelevantFinding"]
    if not choices:
        choices = [l for l in ALL_LABELS if l != "NoRelevantFinding"]
    return rng.choice(choices)


# -------------------------
# Perturbation generators (each returns ONE candidate vector)
# -------------------------
def inter_class_once(orig: Dict[str, int], labels: Set[str], rng: random.Random) -> Optional[Dict[str, int]]:
    # choose a present label with >1 concepts (skip NRF; Cardiomegaly is 1-concept)
    candidates = [l for l in labels if l != "NoRelevantFinding" and len(LABEL_TO_CONCEPTS.get(l, [])) > 1]
    if not candidates:
        return None
    chosen_label = rng.choice(candidates)
    keys = LABEL_TO_CONCEPTS[chosen_label]

    v = deepcopy(orig)

    # randomly choose one of two styles
    if rng.random() < 0.5:
        # toggle one within group
        ones = [k for k in keys if v.get(k, 0) == 1]
        zeros = [k for k in keys if v.get(k, 0) == 0]
        if ones:
            v[rng.choice(ones)] = 0
            if zeros:
                v[rng.choice(zeros)] = 1
        else:
            v[rng.choice(keys)] = 1
    else:
        # resample within group
        set_concepts(v, keys, 0)
        random_set_some_to_one(v, keys, 1, min(2, len(keys)), rng)

    return v


def insertion_once(orig: Dict[str, int], labels: Set[str], rng: random.Random) -> Dict[str, int]:
    current = set(labels)
    other_label = pick_other_label(current, rng)
    keys = LABEL_TO_CONCEPTS[other_label]

    v = deepcopy(orig)
    # if original is NoRelevantFinding, clear it first
    if "NoRelevantFinding" in current:
        v["unremarkable"] = 0

    random_set_some_to_one(v, keys, 1, min(2, len(keys)), rng)
    return v


def deletion_once(orig: Dict[str, int], labels: Set[str], rng: random.Random) -> Optional[Dict[str, int]]:
    # rule: skip deletion for NRF-only cases
    if labels == {"NoRelevantFinding"}:
        return None

    non_nrf = [l for l in labels if l != "NoRelevantFinding"]
    v = deepcopy(orig)

    if len(non_nrf) >= 2:
        # delete one label's concept group
        to_del = rng.choice(non_nrf)
        set_concepts(v, LABEL_TO_CONCEPTS[to_del], 0)
        v["unremarkable"] = 0
        return v

    if len(non_nrf) == 1:
        # delete the single label, replace with NRF
        to_del = non_nrf[0]
        set_concepts(v, LABEL_TO_CONCEPTS[to_del], 0)
        clear_all_non_unremarkable(v)
        v["unremarkable"] = 1
        return v

    return None


def generate_unique(
    make_one_fn,
    orig: Dict[str, int],
    labels: Set[str],
    rng: random.Random,
    max_needed: int,
    max_tries: int = 40,
) -> List[Dict[str, int]]:
    """
    Collect up to max_needed vectors such that:
      - each is unique within this type
      - each is different from the ORIGINAL vector
    """
    outs: List[Dict[str, int]] = []
    seen = set()

    # prevent returning original vector
    seen.add(vec_signature(orig))

    tries = 0
    while len(outs) < max_needed and tries < max_tries:
        tries += 1
        cand = make_one_fn(orig, labels, rng)
        if cand is None:
            break
        cand = ensure_all_keys_present(cand)
        sig = vec_signature(cand)
        if sig in seen:
            continue
        seen.add(sig)
        outs.append(cand)

    return outs


# -------------------------
# Printing helpers (labels + active concepts)
# -------------------------
def print_example_block(title: str, items, max_n: int):
    print(f"\n--- {title} (showing up to {max_n}) ---")
    if not items:
        print("(none)")
        return

    for img, orig, pert_list in items:
        print(f"\nImage: {img}")
        print(f"  Original labels:   {pretty_labels(orig)}")
        print(f"  Original concepts: {active_concepts(orig)}")

        for i, pv in enumerate(pert_list):
            print(f"\n  Perturb {i+1} labels:   {pretty_labels(pv)}")
            print(f"  Perturb {i+1} concepts: {active_concepts(pv)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_file", type=str, default="train_test_split.txt",
                    help="File with lines: image label split")
    ap.add_argument("--concepts_pkl", type=str, default="undersampled_image_vector_pairs.pkl",
                    help="Pickle containing image->concept vectors (tuple structure).")
    ap.add_argument("--out_pkl", type=str, default="test_concept_perturbations.pkl",
                    help="Output pickle file.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print_examples", type=int, default=3)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    split_file = Path(args.split_file)
    concepts_pkl = Path(args.concepts_pkl)
    out_pkl = Path(args.out_pkl)

    test_imgs = load_test_filenames(split_file)
    concept_map = load_concept_map(concepts_pkl)

    missing = [img for img in test_imgs if img not in concept_map]
    if missing:
        print(f"[WARN] {len(missing)} test images missing in concept pickle. Example: {missing[0]}")
    present = [img for img in test_imgs if img in concept_map]
    print(f"Test images: total={len(test_imgs)} present_in_pkl={len(present)} missing={len(missing)}")

    # Output pickle: dict keyed by filename
    out: Dict[str, Dict[str, Any]] = {}

    examples = {"inter_class": [], "insertion": [], "deletion": []}

    for img in sorted(present):
        orig_vec = ensure_all_keys_present(concept_map[img])
        true_labels = set(get_label_from_vector(orig_vec))

        # Inter-class: up to 2 unique (or 0 if cannot)
        inter = generate_unique(inter_class_once, orig_vec, true_labels, rng, max_needed=2)
        # Insertion: up to 2 unique
        ins = generate_unique(insertion_once, orig_vec, true_labels, rng, max_needed=2)

        # Deletion rules:
        #  - 0 if NRF-only
        #  - 1 if exactly one non-NRF label (replacement with NRF -> can't be uniquely duplicated)
        #  - 2 if >=2 non-NRF labels
        if true_labels == {"NoRelevantFinding"}:
            dele = []
        else:
            non_nrf = [l for l in true_labels if l != "NoRelevantFinding"]
            max_del = 1 if len(non_nrf) == 1 else 2
            dele = generate_unique(deletion_once, orig_vec, true_labels, rng, max_needed=max_del)

        # Optional mild warnings for inter/insertion only
        if len(inter) == 1:
            print(f"[WARN] {img}: only 1 unique inter_class perturbation found (diff from original enforced).")
        if len(ins) == 1:
            print(f"[WARN] {img}: only 1 unique insertion perturbation found (diff from original enforced).")

        out[img] = {
            "original_vector": orig_vec,
            "original_labels": sorted(true_labels),
            "perturbations": {
                "inter_class": inter,
                "insertion": ins,
                "deletion": dele,  # empty if NRF-only, or 1 if single-label->NRF
            }
        }

        # Collect a few examples for printing
        if len(examples["inter_class"]) < args.print_examples and inter:
            examples["inter_class"].append((img, orig_vec, inter))
        if len(examples["insertion"]) < args.print_examples and ins:
            examples["insertion"].append((img, orig_vec, ins))
        if len(examples["deletion"]) < args.print_examples and dele:
            examples["deletion"].append((img, orig_vec, dele))

    with out_pkl.open("wb") as f:
        pickle.dump(out, f)

    print(f"\nSaved perturbations pickle: {out_pkl} (N={len(out)})")

    print_example_block("INTER-CLASS perturbations", examples["inter_class"], args.print_examples)
    print_example_block("INSERTION perturbations", examples["insertion"], args.print_examples)
    print_example_block("DELETION perturbations", examples["deletion"], args.print_examples)


if __name__ == "__main__":
    main()
