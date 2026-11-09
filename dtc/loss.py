"""DTC — Differentiable Transition Consistency (paper Sec. 4.2).

EQUATION, over same-KC practice pairs (t, t') with earlier response r_t:
    m    = (2 r_t - 1) * ( logit(p_t') - logit(p_t) )
    DTC  = mean over pairs of  sigmoid( m / tau ),      tau = 0.25
    property = -DTC      (post_train MINIMIZES lam * property)

A pair (t, t') is two CONSECUTIVE opportunities on the same KC inside one
window, at most MAX_GAP steps apart — the pair set of the TCR metrics.

EVAL: HTCR / STCR (Bai et al., AIED 2026) from
evaluation/transition_consistency.py.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from transition_consistency import evaluate_transitions      # noqa: E402

TAU = 0.25        # temperature of the logistic credit
MAX_GAP = 200     # maximum step gap of a same-KC practice pair


def same_kc_pairs(concept_ids, responses, n_valid_steps):
    """Flat tensors over ALL windows of consecutive same-KC practice pairs:
    (window_idx, earlier_step, later_step, earlier_response)."""
    window_idx, earlier_step, later_step, earlier_response = [], [], [], []
    for i in range(len(concept_ids)):
        n_valid = int(n_valid_steps[i])
        window_concepts = concept_ids[i, :n_valid].tolist()
        window_responses = responses[i, :n_valid].tolist()
        last_seen = {}                       # KC id -> its latest step
        for t in range(n_valid):
            kc = window_concepts[t]
            if kc in last_seen and t - last_seen[kc] <= MAX_GAP:
                window_idx.append(i)
                earlier_step.append(last_seen[kc])
                later_step.append(t)
                earlier_response.append(window_responses[last_seen[kc]])
            last_seen[kc] = t
    return (torch.tensor(window_idx), torch.tensor(earlier_step),
            torch.tensor(later_step), torch.tensor(earlier_response))


def dtc(p_correct, pairs, in_batch_mask):
    """DTC of the batch: mean logistic credit over the same-KC pairs whose
    window is in the batch.

    p_correct     (n_batch_windows, T) predictions, rows in ascending
                  window order
    pairs         same_kc_pairs(...) over ALL training windows
    in_batch_mask (n_train_windows,) bool: which windows are in the batch
    """
    window_idx, earlier_step, later_step, earlier_response = pairs
    in_batch = in_batch_mask[window_idx]
    window_idx = window_idx[in_batch]
    earlier_step, later_step = earlier_step[in_batch], later_step[in_batch]
    earlier_response = earlier_response[in_batch].to(p_correct.device)
    # global window index -> row position inside the batch
    batch_row = torch.zeros(in_batch_mask.shape[0], dtype=torch.long,
                            device=p_correct.device)
    batch_row[in_batch_mask] = torch.arange(int(in_batch_mask.sum()),
                                            device=p_correct.device)
    p_earlier = p_correct[batch_row[window_idx], earlier_step] \
        .clamp(1e-4, 1 - 1e-4).double()
    p_later = p_correct[batch_row[window_idx], later_step] \
        .clamp(1e-4, 1 - 1e-4).double()
    direction = (2 * earlier_response - 1).double()
    margin = direction * (torch.logit(p_later) - torch.logit(p_earlier))
    credit = torch.sigmoid(margin / TAU)
    return credit.sum() / max(credit.numel(), 1)


class DTCTerm:
    def __init__(self, max_gap=MAX_GAP):
        self.max_gap = max_gap
        self.pairs = None

    def setup(self, train_question_ids, train_concept_ids, train_responses,
              train_n_valid_steps, train_is_repeat):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Precompute, once over ALL training windows, the flat tensors of
        # consecutive same-KC practice pairs. value() then selects the
        # subset whose window is in the current batch instead of
        # re-scanning sequences.
        self.pairs = tuple(t.to(device) for t in same_kc_pairs(
            train_concept_ids, train_responses, train_n_valid_steps))
        print(f"DTC: {len(self.pairs[0])} training pairs")

    def value(self, p_correct, scored_steps_mask, batch_ctx):
        in_batch_mask = batch_ctx["in_batch_mask"].to(p_correct.device)
        # Negated because the trainer MINIMIZES the property while DTC is
        # a maximize-me score.
        return -dtc(p_correct, self.pairs, in_batch_mask)

    def eval_metrics(self, scored_windows):
        # scored_windows hold (concepts, responses, probs) per window, but
        # evaluate_transitions expects (concepts, probs, responses) — swap
        # the last two fields.
        tcr = evaluate_transitions(
            [(window_concepts, window_probs, window_responses)
             for window_concepts, window_responses, window_probs
             in scored_windows],
            self.max_gap)
        return dict(htcr=tcr.htcr, stcr=tcr.stcr)
