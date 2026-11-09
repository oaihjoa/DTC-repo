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

Only the head trains (lr 1e-3, the small-head convention). The paper sweeps
lam over {0.03125, 0.125, 0.5, 2, 8, 32, 128, 512} at mu = 1. Every metric
(AUC/NLL/fkl/HTCR/STCR) comes from the shared aligned evaluation, so this
baseline lands on the same axes as every weight-space method.

Run: python baselines/frappe_head.py --csv ... --model qikt --ckpt <base> \
         --lam 2 --mu 1 --out ckpt/frappe_L2.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from eval_common import (DEVICE, T_MAX, load_windows, make_dcur,
                         build_model)                        # noqa: E402
from train.post_train_common import evaluate                 # noqa: E402
from dtc.loss import DTCTerm                                 # noqa: E402


class FrappeHead(nn.Module):
    """Frozen base + trainable additive logit correction.

    forward(dcur) returns corrected P(correct), so eval_common's
    collect_scored_windows and every metric run on it unchanged."""

    def __init__(self, base_model, feature_dim, hidden_dim=64):
        super().__init__()
        self.base_model = base_model
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        self.head = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, 1))
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--dataset", default="peiyou")
    ap.add_argument("--model-cfg", default="")
    ap.add_argument("--ckpt", required=True, help="frozen base ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--mu", type=float, default=2.0,
                    help="penalty on mean dz^2 (the sweep knob)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=0)
    ap.add_argument("--model", default="qikt",
                    help="base architecture; its adapter must expose "
                         "predict_with_features + frappe_feature_dim "
                         "(dkt, dkvmn, deep_irt, qikt, ukt)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    (train_question_ids, train_concept_ids, train_responses,
     train_n_valid_steps, train_is_repeat) = load_windows(
        args.csv, [0, 1, 2, 3], min_len=8, with_repeats=True)
    (eval_question_ids, eval_concept_ids, eval_responses,
     eval_n_valid_steps, eval_is_repeat) = load_windows(
        args.csv, [4], min_len=3, with_repeats=True)
    if args.n_train:
        (train_question_ids, train_concept_ids, train_responses,
         train_n_valid_steps, train_is_repeat) = (
            t[:args.n_train] for t in
            (train_question_ids, train_concept_ids, train_responses,
             train_n_valid_steps, train_is_repeat))
    if args.n_eval:
        (eval_question_ids, eval_concept_ids, eval_responses,
         eval_n_valid_steps, eval_is_repeat) = (
            t[:args.n_eval] for t in
            (eval_question_ids, eval_concept_ids, eval_responses,
             eval_n_valid_steps, eval_is_repeat))
    eval_data = (eval_question_ids, eval_concept_ids, eval_responses,
                 eval_n_valid_steps, eval_is_repeat)

    base_model = build_model(args.model, args.dataset, args.ckpt,
                             args.model_cfg)
    base_model.eval()
    feature_dim = base_model.frappe_feature_dim
    model = FrappeHead(base_model, feature_dim, args.hidden).to(DEVICE)
    model.eval()      # convention parity: eval mode, grads flow (head)

    property_term = DTCTerm()
    property_term.setup(train_question_ids, train_concept_ids,
                        train_responses, train_n_valid_steps,
                        train_is_repeat)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    metrics_at_start = evaluate(model, eval_data, property_term,
                                base_model=base_model)
    print(f"iter 0 baseline: {json.dumps(metrics_at_start)}")
    log_file = open(out.with_suffix(".metrics.jsonl"), "w")
    log_file.write(json.dumps({"iter": 0, **metrics_at_start}) + "\n")

    step_positions = torch.arange(T_MAX, device=DEVICE)[None, :]
    t_start = time.time()
    metrics = metrics_at_start
    for it in range(1, args.iters + 1):
        sampled = torch.randint(len(train_question_ids), (args.batch,))
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
        batch_ctx = dict(question_ids=batch_question_ids,
                         concept_ids=batch_concept_ids,
                         responses=batch_responses,
                         n_valid_steps=batch_n_valid,
                         window_indices=window_indices,
                         in_batch_mask=in_batch_mask, model=model)
        property_value = property_term.value(p_correct, scored_steps_mask,
                                             batch_ctx)
        correction_penalty = (model.last_dz.double() ** 2
                              )[scored_steps_mask].mean()
        loss = args.lam * property_value + args.mu * correction_penalty
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        if it % args.eval_every == 0 or it == args.iters:
            metrics = evaluate(model, eval_data, property_term,
                               base_model=base_model)
            print(f"iter {it:>5}  prop={float(property_value):.5f} "
                  f"pen={float(correction_penalty):.5f}  "
                  f"{json.dumps(metrics)} "
                  f"[{(time.time()-t_start)/it*1000:.0f} ms/iter]")
            log_file.write(json.dumps(
                {"iter": it, "prop": float(property_value),
                 "dz2": float(correction_penalty), **metrics}) + "\n")
            log_file.flush()

    torch.save({name: p for name, p in model.head.state_dict().items()},
               out)
    out.with_suffix(".final.json").write_text(json.dumps(
        {"arm": "frappe_head", "args": vars(args),
         "baseline": metrics_at_start, "final": metrics}, indent=2))
    print(f"done in {(time.time()-t_start)/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
