"""Soft equal-opportunity gap property term (fairness arm).

SOURCE: equalized odds / equal opportunity — Hardt, Price, Srebro,
NeurIPS 2016 (arXiv:1610.02413). The threshold-free soft surrogate is this
study's; the objective SHAPE (fairness loss + divergence-to-frozen-base)
matches FRAPPE (ICML'24, arXiv:2312.02592).

EQUATION over masked NON-REPEAT r=1 rows of the two labeled groups A/B:
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


class EqOppTerm:
    def __init__(self, args):
        self.args = args
        self.groups_path = args.groups
        self.attr = args.group_attr

    def _group_tensor(self, student_ids):
        """Per-window 0/1 group label (or -1 = unlabeled/excluded)."""
        gmap = {int(k): int(v) for k, v in
                json.loads(Path(self.groups_path).read_text())[self.attr].items()}
        vals = sorted({v for v in gmap.values() if v >= 0})
        assert len(vals) == 2, f"need 2 groups for {self.attr}, got {vals}"
        group_labels = torch.tensor(
            [{vals[0]: 0, vals[1]: 1}.get(gmap.get(int(u), -1), -1)
             for u in student_ids])
        assert (group_labels == 0).any() and (group_labels == 1).any(), \
            "empty group after join"
        return group_labels

    def setup(self, train_question_ids, train_concept_ids, train_responses,
              train_n_valid_steps, train_is_repeat):
        # reload uids with the IDENTICAL filter to get row-aligned labels
        from eval_common import load_windows
        a = self.args
        *_, student_ids = load_windows(a.csv, [0, 1, 2, 3], min_len=8,
                                       with_uid=True)
        if a.n_train:
            student_ids = student_ids[:a.n_train]
        assert len(student_ids) == len(train_question_ids), \
            "uid/window row mismatch"
        self.train_groups = self._group_tensor(student_ids)
        self.train_is_repeat = train_is_repeat
        self._eval_groups = None

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

    def eval_metrics(self, scored_windows, model=None, eval_data=None):
        if self._eval_groups is None:
            from eval_common import load_windows
            a = self.args
            *_, student_ids = load_windows(a.csv, [4], min_len=3,
                                           with_uid=True)
            if a.n_eval:
                student_ids = student_ids[:a.n_eval]
            self._eval_groups = self._group_tensor(student_ids)
        labels, probs, group_ids = [], [], []
        for (window_concepts, window_responses, window_probs), group in zip(
                scored_windows, self._eval_groups):
            labels.extend(window_responses)
            probs.extend(window_probs)
            group_ids.extend([int(group)] * len(window_responses))
        labels = np.array(labels, float)
        probs, group_ids = np.array(probs), np.array(group_ids)
        mean = lambda x: float(x.mean()) if len(x) else float("nan")
        in_group_a, in_group_b = group_ids == 0, group_ids == 1
        return dict(eo=abs(mean(probs[in_group_a & (labels == 1)])
                           - mean(probs[in_group_b & (labels == 1)])))
