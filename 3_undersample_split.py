import os
import ast
import random
import shutil
from collections import Counter, defaultdict

IMAGES_TXT = "unfiltered_images.txt"
LABELS_TXT = "unfiltered_image_class_labels.txt"
SRC_DIR = "filtered_images"
OUT_DIR = "undersampled_images"

SEED = 42
NORMALS_PER_LARGEST_ABNORMAL = 2  # 2:1 (normals : largest abnormal label)

def load_image_map(path):
    id_to_filename = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, fn = line.split(maxsplit=1)
            id_to_filename[int(idx)] = fn
    return id_to_filename

def load_label_map(path):
    id_to_labels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, lab_str = line.split(maxsplit=1)
            id_to_labels[int(idx)] = ast.literal_eval(lab_str)
    return id_to_labels

def is_normal(labs):
    return labs == ["NoRelevantFinding"]

def main():
    random.seed(SEED)

    id_to_fn = load_image_map(IMAGES_TXT)
    id_to_labels = load_label_map(LABELS_TXT)

    common_ids = sorted(set(id_to_fn) & set(id_to_labels))

    empty_ids = []
    normal_ids = []
    abnormal_ids = []

    # count abnormal label occurrences over ABNORMAL images only
    abnormal_label_counts = Counter()

    for idx in common_ids:
        labs = id_to_labels[idx]

        if labs == []:
            empty_ids.append(idx)
            continue

        if is_normal(labs):
            normal_ids.append(idx)
        else:
            abnormal_ids.append(idx)
            for lab in labs:
                if lab != "NoRelevantFinding":
                    abnormal_label_counts[lab] += 1

    if not abnormal_label_counts:
        raise RuntimeError("No abnormal labels found (after dropping []).")

    largest_abnormal_label, largest_abnormal_count = abnormal_label_counts.most_common(1)[0]

    # target normals = 2 * (largest abnormal label count)
    target_normals = NORMALS_PER_LARGEST_ABNORMAL * largest_abnormal_count
    target_normals = min(target_normals, len(normal_ids))  # cannot exceed available normals
    sampled_normal_ids = random.sample(normal_ids, target_normals)

    selected_ids = sorted(sampled_normal_ids + abnormal_ids)

    os.makedirs(OUT_DIR, exist_ok=True)

    out_images_txt = os.path.join(OUT_DIR, "images.txt")
    out_labels_txt = os.path.join(OUT_DIR, "labels.txt")

    # counts in the FINAL selected set
    final_label_counts = Counter()

    with open(out_images_txt, "w", encoding="utf-8") as f_img, \
         open(out_labels_txt, "w", encoding="utf-8") as f_lab:

        for new_i, old_i in enumerate(selected_ids):
            fn = id_to_fn[old_i]
            src = os.path.join(SRC_DIR, fn)
            dst = os.path.join(OUT_DIR, fn)

            if not os.path.exists(src):
                raise FileNotFoundError(f"Missing image: {src}")

            shutil.copy2(src, dst)

            f_img.write(f"{new_i} {fn}\n")
            f_lab.write(f"{new_i} {id_to_labels[old_i]}\n")

            for lab in id_to_labels[old_i]:
                final_label_counts[lab] += 1

    # -------- stats --------
    print("=== Undersampling complete ===")
    print(f"Total IDs in both files: {len(common_ids)}")
    print(f"Empty-label images dropped ([]): {len(empty_ids)}")
    print(f"Abnormal kept (all): {len(abnormal_ids)}")
    print(f"Normal available: {len(normal_ids)}")

    print("\nLargest abnormal label (counted on abnormal images):")
    print(f"  {largest_abnormal_label}: {largest_abnormal_count}")
    print(f"Target normals for {NORMALS_PER_LARGEST_ABNORMAL}:1 vs largest abnormal label: {target_normals}")

    print(f"\nNormal sampled: {len(sampled_normal_ids)}")
    print(f"Final dataset size: {len(selected_ids)}")

    print("\nLabel counts in FINAL selected set (multi-label occurrences):")
    for k, v in final_label_counts.most_common():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
