"""Equal-opportunity gap of a checkpoint.   (dataset csv x ckpt) -> score

Metric source: equal opportunity, Hardt et al. NeurIPS 2016
(arXiv:1610.02413), in its threshold-free form over kept fold-4 rows:
  eo = | mean(p | r=1, A) - mean(p | r=1, B) |
Requires per-student groups: data/groups_nips_task34.json built by
eo/join_metadata.py.

Run:  python evaluation/evaluation_eo.py --csv ... --ckpt ... --model ... \
          --dataset nips_task34 --groups data/groups_nips_task34.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_common import (MODEL_DEFAULTS, DATASET_CFG, load_windows,
                         build_model, collect_scored_windows)  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True, choices=tuple(MODEL_DEFAULTS))
    ap.add_argument("--dataset", default="nips_task34",
                    choices=tuple(DATASET_CFG))
    ap.add_argument("--model-cfg", default="", help="JSON cfg overrides")
    ap.add_argument("--groups", required=True)
    ap.add_argument("--group-attr", default="gender")
    ap.add_argument("--n-eval", type=int, default=0, help="cap windows (smoke)")
    args = ap.parse_args()

    (question_ids, concept_ids, responses, n_valid_steps, is_repeat,
     student_ids) = load_windows(args.csv, [4], min_len=3,
                                 with_repeats=True, with_uid=True)
    if args.n_eval:
        (question_ids, concept_ids, responses, n_valid_steps, is_repeat,
         student_ids) = (t[:args.n_eval]
                         for t in (question_ids, concept_ids, responses,
                                   n_valid_steps, is_repeat, student_ids))
    # student id -> raw group code for the chosen attribute (e.g. gender)
    group_of_student = {
        int(k): int(v) for k, v in
        json.loads(Path(args.groups).read_text())[args.group_attr].items()}
    # the two smallest non-negative codes become groups A (=0) and B (=1);
    # unknown students / other codes map to -1 and are dropped by the masks
    group_values = sorted({v for v in group_of_student.values() if v >= 0})
    window_group = torch.tensor(
        [{group_values[0]: 0, group_values[1]: 1}.get(
            group_of_student.get(int(sid), -1), -1)
         for sid in student_ids])

    model = build_model(args.model, args.dataset, args.ckpt, args.model_cfg)
    scored_windows = collect_scored_windows(model, question_ids, concept_ids,
                                            responses, n_valid_steps,
                                            is_repeat=is_repeat)
    # flatten kept rows into pooled arrays, tagging every row with the group
    # label of the window (= student) it came from
    labels, probs, group_labels = [], [], []
    for (_concept_ids, window_responses, window_probs), group in zip(
            scored_windows, window_group):
        labels.extend(window_responses)
        probs.extend(window_probs)
        group_labels.extend([int(group)] * len(window_responses))
    labels, probs, group_labels = (np.array(labels, float), np.array(probs),
                                   np.array(group_labels))
    # boolean row masks for the two audited groups (-1 rows fall in neither)
    in_group_a, in_group_b = group_labels == 0, group_labels == 1
    # mean that tolerates an empty slice (a group may lack one class)
    mean = lambda x: float(x.mean()) if len(x) else float("nan")
    # threshold-free equal-opportunity gap: mean score on rows the student
    # actually got RIGHT, group A vs group B
    print(json.dumps({
        "eo": abs(mean(probs[in_group_a & (labels == 1)])
                  - mean(probs[in_group_b & (labels == 1)])),
        "n_a": int(in_group_a.sum()), "n_b": int(in_group_b.sum())}))


if __name__ == "__main__":
    main()
