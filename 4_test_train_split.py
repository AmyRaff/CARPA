import os
import ast
import random
import shutil
from collections import Counter, defaultdict

# ---- INPUT: your undersampled dataset outputs ----
IN_DIR = "undersampled_images"  # contains images + images.txt + labels.txt
IMAGES_TXT = os.path.join(IN_DIR, "images.txt")  # lines: <id> <filename>
LABELS_TXT = os.path.join(IN_DIR, "labels.txt")  # lines: <id> <labels_list>

# ---- OUTPUT ----
OUT_DIR = "split_dataset"  # will create train/val/test
COPY_IMAGES = True         # set False to only write manifests (no copying)

SEED = 42
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

# ---- Repair controls ----
REPAIR_ENABLE = True
# Ensure each label appears at least this many times in each split (when feasible).
# Use 1 to guarantee presence. If you want stronger, set 3/5/etc.
MIN_PER_SPLIT_PER_LABEL = 1
# Don't try to enforce MIN_PER_SPLIT_PER_LABEL for labels with fewer than this many total occurrences.
# (e.g. a label seen only once can't appear in all splits)
MIN_GLOBAL_TO_ENFORCE = 3
# Limit how hard we try to repair each label/split (prevents long runtimes)
MAX_SWAPS_PER_LABEL_SPLIT = 25

def load_image_map(path):
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, fn = line.split(maxsplit=1)
            m[int(idx)] = fn
    return m

def load_label_map(path):
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, lab_str = line.split(maxsplit=1)
            m[int(idx)] = ast.literal_eval(lab_str)
    return m

def stable_round_targets(total, ratios, split_names):
    """
    Returns per-split integer targets summing to total.
    Largest-remainder method.
    """
    raw = {s: total * ratios[s] for s in split_names}
    base = {s: int(raw[s]) for s in split_names}
    remainder = total - sum(base.values())
    fracs = sorted(split_names, key=lambda s: (raw[s] - base[s]), reverse=True)
    for i in range(remainder):
        base[fracs[i % len(fracs)]] += 1
    return base

def main():
    random.seed(SEED)

    id_to_fn = load_image_map(IMAGES_TXT)
    id_to_labels = load_label_map(LABELS_TXT)

    ids = sorted(set(id_to_fn) & set(id_to_labels))
    if not ids:
        raise RuntimeError("No overlapping IDs between images.txt and labels.txt")

    # Drop any [] that might still exist
    dropped_empty = [i for i in ids if id_to_labels[i] == []]
    ids = [i for i in ids if id_to_labels[i] != []]

    # Global label counts (multi-label occurrences)
    global_label_counts = Counter()
    for i in ids:
        for lab in id_to_labels[i]:
            global_label_counts[lab] += 1

    split_names = list(SPLIT_RATIOS.keys())
    total_n = len(ids)

    # Desired sample counts per split
    target_n = stable_round_targets(total_n, SPLIT_RATIOS, split_names)

    # Desired label counts per split (proportional)
    target_label_counts = {s: {} for s in split_names}
    for lab, total_lab_cnt in global_label_counts.items():
        per_split = stable_round_targets(total_lab_cnt, SPLIT_RATIOS, split_names)
        for s in split_names:
            target_label_counts[s][lab] = per_split[s]

    # Current counts
    current_n = {s: 0 for s in split_names}
    current_label_counts = {s: Counter() for s in split_names}

    # Remaining label counts across unassigned ids
    remaining_label_counts = Counter(global_label_counts)

    # For faster lookup: ids that contain each label
    ids_by_label = defaultdict(list)
    for i in ids:
        for lab in id_to_labels[i]:
            ids_by_label[lab].append(i)

    for lab in ids_by_label:
        random.shuffle(ids_by_label[lab])

    assignment = {}
    unassigned = set(ids)

    def cap_left(s):
        return target_n[s] - current_n[s]

    def choose_rarest_label():
        rare_label = None
        rare_count = None
        for lab, cnt in remaining_label_counts.items():
            if cnt <= 0:
                continue
            if rare_count is None or cnt < rare_count:
                rare_label, rare_count = lab, cnt
        return rare_label

    def score_split_for_sample(s, sample_labels, focus_label):
        """
        True iterative stratification scoring:
        1) focus (rarest) label deficit dominates
        2) then other label deficits
        3) tiny tie-break by remaining capacity
        """
        if cap_left(s) <= 0:
            return -10**18

        focus_def = target_label_counts[s][focus_label] - current_label_counts[s][focus_label]
        if focus_def < 0:
            focus_def = 0

        other_def = 0
        for lab in sample_labels:
            if lab == focus_label:
                continue
            d = target_label_counts[s][lab] - current_label_counts[s][lab]
            if d > 0:
                other_def += d

        return (10_000 * focus_def) + other_def + (0.001 * cap_left(s))

    # ---- Iterative assignment ----
    while unassigned:
        focus = choose_rarest_label()
        if focus is None:
            # distribute remaining by capacity (rare)
            rest = list(unassigned)
            random.shuffle(rest)
            for i in rest:
                s = max(split_names, key=lambda x: cap_left(x))
                assignment[i] = s
                current_n[s] += 1
                for lab in id_to_labels[i]:
                    current_label_counts[s][lab] += 1
                unassigned.remove(i)
            break

        # pick an unassigned sample that contains the focus label
        candidate = None
        for i in ids_by_label[focus]:
            if i in unassigned:
                candidate = i
                break
        if candidate is None:
            remaining_label_counts[focus] = 0
            continue

        labs = id_to_labels[candidate]
        scores = {s: score_split_for_sample(s, labs, focus) for s in split_names}
        best = max(scores.values())
        best_splits = [s for s in split_names if scores[s] == best]
        chosen = random.choice(best_splits)

        assignment[candidate] = chosen
        current_n[chosen] += 1
        for lab in labs:
            current_label_counts[chosen][lab] += 1
            remaining_label_counts[lab] -= 1
        unassigned.remove(candidate)

    # Build split lists
    split_ids = {s: [] for s in split_names}
    for i, s in assignment.items():
        split_ids[s].append(i)
    for s in split_names:
        random.shuffle(split_ids[s])

    # ---- Repair pass ----
    if REPAIR_ENABLE:
        split_of = {}
        for s in split_names:
            for i in split_ids[s]:
                split_of[i] = s

        # Track label counts per split (initialize from current_label_counts)
        # current_label_counts already has them, but keep it as the single source of truth.

        # Helper: compute "cost" of removing/adding sample relative to target deficits
        def delta_score_for_move(sample_id, src, dst):
            """
            Negative is good (reduces total absolute error from label targets),
            positive is bad (increases error).
            """
            labs = id_to_labels[sample_id]

            def abs_err(s, lab, new_count):
                return abs(target_label_counts[s][lab] - new_count)

            before = 0
            after = 0

            # sample count error (soft)
            before += abs(target_n[src] - current_n[src]) + abs(target_n[dst] - current_n[dst])
            after += abs(target_n[src] - (current_n[src] - 1)) + abs(target_n[dst] - (current_n[dst] + 1))

            for lab in labs:
                before += abs_err(src, lab, current_label_counts[src][lab])
                before += abs_err(dst, lab, current_label_counts[dst][lab])

                after += abs_err(src, lab, current_label_counts[src][lab] - 1)
                after += abs_err(dst, lab, current_label_counts[dst][lab] + 1)

            return after - before

        def apply_move(sample_id, src, dst):
            # remove from src list, add to dst list
            split_ids[src].remove(sample_id)
            split_ids[dst].append(sample_id)
            split_of[sample_id] = dst

            current_n[src] -= 1
            current_n[dst] += 1
            for lab in id_to_labels[sample_id]:
                current_label_counts[src][lab] -= 1
                current_label_counts[dst][lab] += 1

        # For each label, ensure it appears in each split (>= MIN_PER_SPLIT_PER_LABEL)
        enforce_labels = [lab for lab, cnt in global_label_counts.items() if cnt >= MIN_GLOBAL_TO_ENFORCE]

        # Pre-index samples by label for faster candidate selection
        samples_with_label = {lab: set(ids_by_label[lab]) for lab in enforce_labels}

        for lab in enforce_labels:
            for dst in split_names:
                needed = MIN_PER_SPLIT_PER_LABEL - current_label_counts[dst][lab]
                swaps_done = 0

                while needed > 0 and swaps_done < MAX_SWAPS_PER_LABEL_SPLIT:
                    # find a source split that has extra of this label
                    src_candidates = [s for s in split_names if s != dst and current_label_counts[s][lab] > MIN_PER_SPLIT_PER_LABEL]
                    if not src_candidates:
                        break

                    # pick best move (or best swap) from any source
                    best_action = None  # ("move", sample_id, src, dst) or ("swap", a, src, b, dst)
                    best_cost = float("inf")

                    for src in src_candidates:
                        # candidate sample in src containing lab
                        cand_ids = [i for i in split_ids[src] if lab in id_to_labels[i]]
                        random.shuffle(cand_ids)
                        cand_ids = cand_ids[:200]  # cap search for speed

                        for a in cand_ids:
                            # Direct move if dst has capacity; otherwise do swap
                            if current_n[dst] < target_n[dst]:
                                cost = delta_score_for_move(a, src, dst)
                                if cost < best_cost:
                                    best_cost = cost
                                    best_action = ("move", a, src, dst)
                            else:
                                # swap with some b from dst that does NOT have lab (prefer)
                                b_candidates = [j for j in split_ids[dst] if lab not in id_to_labels[j]]
                                if not b_candidates:
                                    b_candidates = list(split_ids[dst])
                                random.shuffle(b_candidates)
                                b_candidates = b_candidates[:200]

                                for b in b_candidates:
                                    # approximate swap cost by move a->dst plus move b->src
                                    cost = delta_score_for_move(a, src, dst) + delta_score_for_move(b, dst, src)
                                    if cost < best_cost:
                                        best_cost = cost
                                        best_action = ("swap", a, src, b, dst)

                    if best_action is None:
                        break

                    # Apply best action
                    if best_action[0] == "move":
                        _, a, src, dst_ = best_action
                        apply_move(a, src, dst_)
                    else:
                        _, a, src, b, dst_ = best_action
                        apply_move(a, src, dst_)
                        apply_move(b, dst_, src)

                    swaps_done += 1
                    needed = MIN_PER_SPLIT_PER_LABEL - current_label_counts[dst][lab]

    # ---- Write outputs (and copy images optionally) ----
    os.makedirs(OUT_DIR, exist_ok=True)

    def write_split(split_name, ids_list):
        split_dir = os.path.join(OUT_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)

        out_images = os.path.join(split_dir, "images.txt")
        out_labels = os.path.join(split_dir, "labels.txt")

        with open(out_images, "w", encoding="utf-8") as f_img, \
             open(out_labels, "w", encoding="utf-8") as f_lab:
            for new_i, old_i in enumerate(ids_list):
                fn = id_to_fn[old_i]

                if COPY_IMAGES:
                    src = os.path.join(IN_DIR, fn)
                    dst = os.path.join(split_dir, fn)
                    if not os.path.exists(src):
                        raise FileNotFoundError(f"Missing image in {IN_DIR}: {src}")
                    shutil.copy2(src, dst)

                f_img.write(f"{new_i} {fn}\n")
                f_lab.write(f"{new_i} {id_to_labels[old_i]}\n")

    for s in split_names:
        random.shuffle(split_ids[s])
        write_split(s, split_ids[s])

    # ---- Report ----
    print("=== Multi-label stratified split complete ===")
    print(f"Total samples (after dropping any []): {len(ids)}")
    if dropped_empty:
        print(f"Dropped [] cases found in input: {len(dropped_empty)}")
    print(f"Targets: {target_n}")
    print(f"Actual:  {{'train': {len(split_ids['train'])}, 'val': {len(split_ids['val'])}, 'test': {len(split_ids['test'])}}}")
    print(f"COPY_IMAGES={COPY_IMAGES}")
    if REPAIR_ENABLE:
        print(f"REPAIR_ENABLE=True (MIN_PER_SPLIT_PER_LABEL={MIN_PER_SPLIT_PER_LABEL}, MIN_GLOBAL_TO_ENFORCE={MIN_GLOBAL_TO_ENFORCE})")

    # Label prevalence check (top labels)
    def label_counts_for_split(s):
        c = Counter()
        for i in split_ids[s]:
            for lab in id_to_labels[i]:
                c[lab] += 1
        return c

    global_top = global_label_counts.most_common(12)
    print("\nTop labels (global):")
    for lab, cnt in global_top:
        print(f"  {lab}: {cnt}")

    for s in split_names:
        c = label_counts_for_split(s)
        print(f"\nTop labels ({s}):")
        for lab, cnt in c.most_common(12):
            share = cnt / global_label_counts[lab] if global_label_counts[lab] else 0.0
            print(f"  {lab}: {cnt}  (share of global {lab}: {share:.3f})")

    # Also explicitly report any label missing from a split
    missing = defaultdict(list)
    for lab in global_label_counts:
        for s in split_names:
            if current_label_counts[s][lab] == 0:
                missing[s].append(lab)
    any_missing = any(missing[s] for s in split_names)
    if any_missing:
        print("\nWARNING: Some labels are still missing in splits (not always feasible):")
        for s in split_names:
            if missing[s]:
                print(f"  {s} missing: {missing[s]}")
    else:
        print("\nAll labels present in all splits (at least once).")

if __name__ == "__main__":
    main()
