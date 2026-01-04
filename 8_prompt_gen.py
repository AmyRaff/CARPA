import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Any


# -------------------------
# Label mapping (unchanged)
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
# Prompt construction (UPDATED)
# -------------------------
NO_FINDING_PROMPT = "Normal chest radiograph."


def construct_prompt_from_labels(labels: List[str]) -> str:
    if not labels or labels == ["NoRelevantFinding"]:
        return NO_FINDING_PROMPT
    return ", ".join(sorted(labels))


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_pkl", type=str, default="test_concept_perturbations.pkl",
                    help="Input pickle with original + perturbation vectors")
    ap.add_argument("--output_pkl", type=str, default="test_radedit_prompts.pkl",
                    help="Output pickle with RadEdit prompts")
    args = ap.parse_args()

    input_path = Path(args.input_pkl)
    output_path = Path(args.output_pkl)

    with input_path.open("rb") as f:
        data = pickle.load(f)

    out: Dict[str, List[Dict[str, Any]]] = {}

    for img, rec in data.items():
        prompt_entries: List[Dict[str, Any]] = []

        for pert_type, vec_list in rec["perturbations"].items():
            for vec in vec_list:
                labels = get_label_from_vector(vec)
                prompt = construct_prompt_from_labels(labels)

                prompt_entries.append({
                    "filename": img,
                    "perturbation_type": pert_type,
                    "prompt": prompt,
                })

        out[img] = prompt_entries

    with output_path.open("wb") as f:
        pickle.dump(out, f)

    print(f"Saved RadEdit prompts to: {output_path}")


if __name__ == "__main__":
    main()
