"""FRAPPE-style output post-processing baseline.

SOURCE: Tifrea et al., "FRAPPE: A Group Fairness Framework for
Post-Processing Everything", ICML 2024 (arXiv:2312.02592). Their recipe:
keep the base model FROZEN and train a small additive correction on the
output logits with the property loss plus a regularizer on the
correction's magnitude. Here the fairness objective is replaced by DTC:

    p(i,t) = sigmoid( z_base(i,t) + dz(i,t) ),
    dz = head(features),  features = the frozen base's causal per-step
         features (history-only state + response-free question identity)
    L  = lam * (-DTC(p)) + mu * mean_{scored}(dz^2)

Only the head trains (lr 1e-3, the small-head convention; hidden width 64,
mu = 1). The paper sweeps lam over {0.03125, 0.125, 0.5, 2, 8, 32, 128,
512}. Every metric (AUC/NLL/fkl/HTCR/STCR) comes from the shared
evaluation, so this baseline lands on the same axes as every other method.

    python baselines/frappe_head.py --model qikt --dataset peiyou \
        --ckpt ckpt/qikt_peiyou.pt --lam 2 --out ckpt/qikt_peiyou_frappe2.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from eval_common import (DEVICE, T_MAX, MODEL_DEFAULTS, DATASET_CFG,
                         make_dcur, build_model)             # noqa: E402
from train.post_train_common import (evaluate, load_train_eval, ITERS,
                                     BATCH, EVAL_EVERY)      # noqa: E402
from dtc.loss import DTCTerm                                 # noqa: E402

LEARNING_RATE = 1e-3
MU = 1.0          # weight of the mean dz^2 penalty
HIDDEN = 64       # width of the correction head


class FrappeHead(nn.Module):
    """Frozen base + trainable additive logit correction.

    forward(dcur) returns corrected P(correct), so eval_common's
    collect_scored_windows and every metric run on it unchanged."""

    def __init__(self, base_model, feature_dim):
        super().__init__()
        self.base_model = base_model
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        self.head = nn.Sequential(nn.Linear(feature_dim, HIDDEN),
                                  nn.ReLU(),
                                  nn.Linear(HIDDEN, 1))
        # zero-init the last layer: dz = 0 at init -> exactly the base
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, dcur):
        with torch.no_grad():
            p_base, features = self.base_model.predict_with_features(dcur)
        z_base = torch.log(p_base.clamp(1e-5, 1 - 1e-5)
                           / (1 - p_base.clamp(1e-5, 1 - 1e-5)))
        dz = self.head(features).squeeze(-1)                  # (B, T)
        self.last_dz = dz
        return torch.sigmoid(z_base + dz)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_DEFAULTS),
                    help="its adapter must expose predict_with_features "
                         "(dkt, dkvmn, deep_irt, qikt, ukt)")
    ap.add_argument("--dataset", required=True, choices=sorted(DATASET_CFG))
    ap.add_argument("--ckpt", required=True, help="frozen base checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, required=True,
                    help="weight of the DTC term (the sweep knob)")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    train_data, eval_data = load_train_eval(args.dataset)
    (train_question_ids, train_concept_ids, train_responses,
     train_n_valid_steps, _train_is_repeat) = train_data

    base_model = build_model(args.model, args.dataset, args.ckpt)
    model = FrappeHead(base_model, base_model.frappe_feature_dim).to(DEVICE)
    model.eval()      # same protocol as the loop: eval mode, grads flow

    property_term = DTCTerm()
    property_term.setup(*train_data)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=LEARNING_RATE)

    metrics_at_start = evaluate(model, eval_data, property_term, base_model)
    print(f"iter 0 baseline: {json.dumps(metrics_at_start)}")

    step_positions = torch.arange(T_MAX, device=DEVICE)[None, :]
    t_start = time.time()
    metrics = metrics_at_start
    for it in range(1, ITERS + 1):
        sampled = torch.randint(len(train_question_ids), (BATCH,))
        in_batch_mask = torch.zeros(len(train_question_ids),
                                    dtype=torch.bool)
        in_batch_mask[sampled] = True
        window_indices = torch.nonzero(in_batch_mask).squeeze(1)
        batch_question_ids = train_question_ids[window_indices].to(DEVICE)
        batch_concept_ids = train_concept_ids[window_indices].to(DEVICE)
        batch_responses = train_responses[window_indices].to(DEVICE)
        batch_n_valid = train_n_valid_steps[window_indices].to(DEVICE)

        p_correct = model(make_dcur(batch_question_ids, batch_concept_ids,
                                    batch_responses))
        scored_steps_mask = ((step_positions >= 1)
                             & (step_positions < batch_n_valid[:, None]))
        property_value = property_term.value(
            p_correct, scored_steps_mask, dict(in_batch_mask=in_batch_mask))
        correction_penalty = (model.last_dz.double() ** 2
                              )[scored_steps_mask].mean()
        loss = args.lam * property_value + MU * correction_penalty
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        if it % EVAL_EVERY == 0 or it == ITERS:
            metrics = evaluate(model, eval_data, property_term, base_model)
            print(f"iter {it:>5}  prop={float(property_value):.5f} "
                  f"pen={float(correction_penalty):.5f}  "
                  f"{json.dumps(metrics)} "
                  f"[{(time.time()-t_start)/it*1000:.0f} ms/iter]")

    torch.save(model.head.state_dict(), out)
    out.with_suffix(".final.json").write_text(json.dumps(
        {"args": vars(args), "baseline": metrics_at_start,
         "final": metrics}, indent=2))
    print(f"done in {(time.time()-t_start)/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
