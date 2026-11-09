"""Property terms vs independent references, on random tensors (CPU, seeded).

  DTC  (dtc/loss.py)   vs an explicit Python loop over same-KC pairs that
       implements the paper's equation directly (Sec. 4.2)
  EO   (eo/loss.py)    vs the direct equal-opportunity gap formula

Run: python tests/test_losses.py    (from the repo root)
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from dtc.loss import DTCTerm, same_kc_pairs, TAU               # noqa: E402
from eo.loss import EqOppTerm                                 # noqa: E402

torch.manual_seed(0)
rng = np.random.default_rng(0)

# ---- synthetic train windows (padded to T; only the first n_valid_steps
# positions of each window are real interactions) ---------------------------
N_WIN, T, N_KC = 24, 60, 12
train_concept_ids = torch.from_numpy(rng.integers(0, N_KC, (N_WIN, T)))
train_responses = torch.from_numpy(rng.integers(0, 2, (N_WIN, T)))
train_n_valid_steps = torch.from_numpy(rng.integers(20, T, N_WIN))
# KC-expansion repeat flags: 1 marks extra rows of a multi-KC question
train_is_repeat = torch.from_numpy(
    (rng.random((N_WIN, T)) < 0.15).astype(np.int64))
train_question_ids = torch.from_numpy(rng.integers(0, 30, (N_WIN, T)))

B = 8
# Simulate one training batch of B distinct windows, matching the trainer's
# convention: in_batch_mask flags membership over ALL train windows, and
# window_indices lists the members sorted, so batch row i of the prediction
# tensor holds window window_indices[i].
in_batch_mask = torch.zeros(N_WIN, dtype=torch.bool)
window_indices = torch.from_numpy(
    rng.choice(N_WIN, B, replace=False)).sort().values
in_batch_mask[window_indices] = True
p_correct = torch.rand(B, T) * 0.9 + 0.05      # away from clamp boundaries
p_correct.requires_grad_(True)
step_positions = torch.arange(T)[None, :]
# score positions 1..n_valid-1: real answers with at least one step of
# history; (1, T) step grid broadcast against (B, 1) per-window lengths
scored_steps_mask = ((step_positions >= 1)
                     & (step_positions
                        < train_n_valid_steps[window_indices][:, None]))
# the batch context every term's value() receives (post_train_common contract)
batch_ctx = dict(question_ids=train_question_ids[window_indices],
                 concept_ids=train_concept_ids[window_indices],
                 responses=train_responses[window_indices],
                 n_valid_steps=train_n_valid_steps[window_indices],
                 window_indices=window_indices,
                 in_batch_mask=in_batch_mask, model=None)

# DTC vs an explicit loop over pairs ---------------------------------------
term = DTCTerm()
term.pairs = same_kc_pairs(train_concept_ids, train_responses,
                           train_n_valid_steps)
v_new = term.value(p_correct, scored_steps_mask, batch_ctx)
(pair_window_idx, pair_earlier_step, pair_later_step,
 pair_earlier_response) = term.pairs
# keep only pairs whose window is in this batch
pair_in_batch = in_batch_mask[pair_window_idx]
assert int(pair_in_batch.sum()) > 0, "no pairs in the synthetic batch"
# global window index -> row position inside the batch prediction tensor
window_to_batch_row = {int(win): row
                       for row, win in enumerate(window_indices)}
assert TAU == 0.25
logit = lambda p: math.log(p / (1 - p))
credits = []
for w, t_early, t_late, r_early in zip(
        pair_window_idx[pair_in_batch], pair_earlier_step[pair_in_batch],
        pair_later_step[pair_in_batch], pair_earlier_response[pair_in_batch]):
    p_earlier = float(p_correct[window_to_batch_row[int(w)], int(t_early)])
    p_later = float(p_correct[window_to_batch_row[int(w)], int(t_late)])
    # after a correct answer the logit should rise, after an incorrect one
    # it should fall; the logistic credit is > 1/2 exactly when it does
    margin = (2 * int(r_early) - 1) * (logit(p_later) - logit(p_earlier))
    credits.append(1 / (1 + math.exp(-margin / TAU)))
dtc_reference = float(np.mean(credits))
assert abs(-float(v_new) - dtc_reference) < 1e-9, \
    f"dtc: {-float(v_new)} != {dtc_reference}"
print(f"  ok dtc: {-float(v_new):.6f} over {len(credits)} pairs")
v_new.backward()
assert p_correct.grad is not None and torch.isfinite(p_correct.grad).all()
assert p_correct.grad.abs().sum() > 0
print("  ok dtc backward (finite, non-zero grads)")
# fresh leaf so the next term builds its graph on an unused tensor
p_correct = p_correct.detach().requires_grad_(True)

# EO gap vs direct formula -------------------------------------------------
# build the term via __new__ (its __init__ would read a groups file) and
# inject synthetic per-window group labels + repeat flags directly
eo = EqOppTerm.__new__(EqOppTerm)
eo.train_groups = torch.from_numpy(rng.integers(0, 2, N_WIN))
eo.train_is_repeat = train_is_repeat
v_new = EqOppTerm.value(eo, p_correct, scored_steps_mask, batch_ctx)
batch_groups = eo.train_groups[window_indices]
non_repeat = train_is_repeat[window_indices] == 0
p_correct_f64 = p_correct.clamp(1e-5, 1 - 1e-5).double()
# equal-opportunity gap: mean prediction on ACTUALLY-correct, scored,
# non-repeat steps, compared between the two groups; the per-window group
# label broadcasts down each row via [:, None]
in_group0_correct = (scored_steps_mask & non_repeat
                     & (batch_ctx["responses"].double() == 1)
                     & (batch_groups[:, None] == 0))
in_group1_correct = (scored_steps_mask & non_repeat
                     & (batch_ctx["responses"].double() == 1)
                     & (batch_groups[:, None] == 1))
v_ref = (p_correct_f64[in_group0_correct].mean()
         - p_correct_f64[in_group1_correct].mean()).abs()
assert torch.equal(v_new, v_ref), f"eo gap: {float(v_new)} != {float(v_ref)}"
print(f"  ok eo gap: {float(v_new):.6f}")

print("PASS: property terms match their references")
