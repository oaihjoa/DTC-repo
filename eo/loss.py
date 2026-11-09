"""Soft equal-opportunity gap property term (nips_task34, gender).

SOURCE: equalized odds / equal opportunity — Hardt, Price, Srebro,
NeurIPS 2016 (arXiv:1610.02413). The threshold-free soft surrogate is
ours; the objective SHAPE (fairness loss + divergence-to-frozen-base)
matches FRAPPE (ICML'24, arXiv:2312.02592).

EQUATION over scored NON-REPEAT r=1 rows of the two labeled groups A/B:
  property = | mean(p | r=1, A) - mean(p | r=1, B) |
(the r=1 branch is ~4:1 dominant in KT — the best-conditioned member of the
EqOdds family).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from eval_common import csv_path, load_windows               # noqa: E402

GROUPS_JSON = REPO_ROOT / "data" / "groups_nips_task34.json"
GROUP_ATTR = "gender"


def window_groups(student_ids):
    """Per-window 0/1 group label (or -1 = unlabeled/excluded)."""
    group_of_student = {
        int(k): int(v) for k, v in
        json.loads(GROUPS_JSON.read_text())[GROUP_ATTR].items()}
    # the two non-negative codes become groups A (=0) and B (=1)
    codes = sorted({v for v in group_of_student.values() if v >= 0})
    assert len(codes) == 2, f"need 2 groups for {GROUP_ATTR}, got {codes}"
    group_labels = torch.tensor(
        [{codes[0]: 0, codes[1]: 1}.get(group_of_student.get(int(u), -1), -1)
         for u in student_ids])
    assert (group_labels == 0).any() and (group_labels == 1).any(), \
        "empty group after join"
    return group_labels


def eo_gap(scored_windows, groups):
    """| mean(p | r=1, A) - mean(p | r=1, B) | over the scored rows; groups
    holds one 0/1/-1 label per window, aligned with scored_windows."""
    labels, probs, group_ids = [], [], []
    for (_concepts, window_responses, window_probs), group in zip(
            scored_windows, groups):
        labels.extend(window_responses)
        probs.extend(window_probs)
        group_ids.extend([int(group)] * len(window_responses))
    labels = np.array(labels, float)
    probs, group_ids = np.array(probs), np.array(group_ids)
    mean = lambda x: float(x.mean()) if len(x) else float("nan")
    return abs(mean(probs[(group_ids == 0) & (labels == 1)])
               - mean(probs[(group_ids == 1) & (labels == 1)]))


class EqOppTerm:
    def __init__(self, args):
        self.csv = csv_path(args.dataset)

    def setup(self, train_question_ids, train_concept_ids, train_responses,
              train_n_valid_steps, train_is_repeat):
        # reload uids with the IDENTICAL filters to get row-aligned labels
        *_, student_ids = load_windows(self.csv, [0, 1, 2, 3], min_len=8,
                                       with_uid=True)
        assert len(student_ids) == len(train_question_ids), \
            "uid/window row mismatch"
        self.train_groups = window_groups(student_ids)
        self.train_is_repeat = train_is_repeat
        *_, student_ids = load_windows(self.csv, [4], min_len=3,
                                       with_uid=True)
        self.eval_groups = window_groups(student_ids)

    def value(self, p_correct, scored_steps_mask, batch_ctx):
        p_correct_f64 = p_correct.clamp(1e-5, 1 - 1e-5).double()
        responses_f64 = batch_ctx["responses"].double()
        batch_groups = self.train_groups[batch_ctx["window_indices"]] \
            .to(p_correct.device)
        non_repeat = self.train_is_repeat[batch_ctx["window_indices"]] \
            .to(p_correct.device) == 0
        in_a = scored_steps_mask & non_repeat & (responses_f64 == 1) \
            & (batch_groups[:, None] == 0)
        in_b = scored_steps_mask & non_repeat & (responses_f64 == 1) \
            & (batch_groups[:, None] == 1)
        if in_a.any() and in_b.any():
            return (p_correct_f64[in_a].mean()
                    - p_correct_f64[in_b].mean()).abs()
        return torch.zeros((), dtype=torch.float64, device=p_correct.device)

    def eval_metrics(self, scored_windows):
        return dict(eo=eo_gap(scored_windows, self.eval_groups))
