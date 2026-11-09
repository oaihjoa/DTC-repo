"""The post-hoc adaptation loop shared by dtc/ and eo/.

THE OBJECTIVE (paper Sec. 4.4):

    L(theta) = lam   * Property(p_theta)                [the loss dir's term]
             + beta  * mean_t D( p_base_t , p_theta_t ) [prediction anchor]
             + gamma * BCE(p_theta, responses)
             + l2sp/2 * || theta - theta_base ||^2

    D defaults to the forward Bernoulli KL, frozen base first:
        KL(Bern(a)||Bern(b)) = a log(a/b) + (1-a) log((1-a)/(1-b))
    closed form per step, no sampling.

A term is on when its weight is given (all default to 0 except lam = 1), so
the preservation mechanisms of the paper (Sec. 5.4) are:
    no anchor   (no flag)
    KL (ours)   --beta B
    reverse KL  --beta B --divergence reverse_kl
    logit-MSE   --beta B --divergence logit_mse
    BCE         --gamma G
    L2-SP       --l2sp A
    LoRA        --gamma 8 --lora-rank R
FRAPPE and WiSE-FT live in baselines/ (they are not trained by this loop).

PROTOCOL (fixed): model.eval() with grads flowing (no dropout), Adam lr
5e-6, grad-clip 1.0, batch = unique of 256 sampled windows, 1500 iters,
eval every 250 on fold 4 (KC-expansion sibling rows are not scored).

INTERFACE for a loss dir:  run_cli(doc, factory) where factory(args)
returns a property term with
    .setup(train_question_ids, train_concept_ids, train_responses,
           train_n_valid_steps, train_is_repeat)  -> None
    .value(p_correct, scored_steps_mask, batch_ctx) -> DIFFERENTIABLE scalar
    .eval_metrics(scored_windows) -> dict of floats
batch_ctx keys: responses (the batch tensor), window_indices (which train
    windows are in the batch, sorted), in_batch_mask (bool over ALL train
    windows).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from eval_common import (DEVICE, T_MAX, MODEL_DEFAULTS, DATASET_CFG,
                         csv_path, load_windows, make_dcur, forward_preds,
                         build_model, collect_scored_windows, flatten, auc,
                         nll)  # noqa: E402

LEARNING_RATE = 5e-6
ITERS = 1500
BATCH = 256
EVAL_EVERY = 250


def anchor_divergence(kind, p_ref, p_now):
    """Per-step divergence between a reference prediction and the current
    one, elementwise over (B, T). The caller detaches p_ref."""
    if kind == "forward_kl":
        return (p_ref * (p_ref / p_now).log()
                + (1 - p_ref) * ((1 - p_ref) / (1 - p_now)).log())
    if kind == "reverse_kl":
        return (p_now * (p_now / p_ref).log()
                + (1 - p_now) * ((1 - p_now) / (1 - p_ref)).log())
    if kind == "logit_mse":
        return (torch.log(p_now / (1 - p_now))
                - torch.log(p_ref / (1 - p_ref))) ** 2
    raise ValueError(f"unknown divergence {kind!r}")


def load_train_eval(dataset):
    """(train tensors, eval tensors): folds 0-3 and fold 4, each as
    (question_ids, concept_ids, responses, n_valid_steps, is_repeat)."""
    csv = csv_path(dataset)
    return (load_windows(csv, [0, 1, 2, 3], min_len=8, with_repeats=True),
            load_windows(csv, [4], min_len=3, with_repeats=True))


def mean_forward_kl(reference_model, model, eval_data):
    """MEASUREMENT (no grads): mean closed-form forward Bernoulli KL
    KL(Bern(p_reference) || Bern(p_model)) over all scored eval steps."""
    (eval_question_ids, eval_concept_ids, eval_responses,
     eval_n_valid_steps, _eval_is_repeat) = eval_data
    fkl_sum, n_steps = 0.0, 0
    step_positions = torch.arange(T_MAX, device=DEVICE)[None, :]
    with torch.no_grad():
        for lo in range(0, len(eval_question_ids), 256):
            bq, bc, br = (t[lo:lo + 256].to(DEVICE) for t in
                          (eval_question_ids, eval_concept_ids,
                           eval_responses))
            bn = eval_n_valid_steps[lo:lo + 256].to(DEVICE)
            mask = (step_positions >= 1) & (step_positions < bn[:, None])
            p_now = forward_preds(model, make_dcur(bq, bc, br)) \
                .clamp(1e-5, 1 - 1e-5).double()
            p_ref = forward_preds(reference_model, make_dcur(bq, bc, br)) \
                .clamp(1e-5, 1 - 1e-5).double()
            fkl_sum += float(anchor_divergence("forward_kl", p_ref, p_now
                                               )[mask].sum())
            n_steps += int(mask.sum())
    return fkl_sum / max(n_steps, 1)


def evaluate(model, eval_data, property_term, base_model):
    """Fold-4 evaluation: AUC, NLL, fkl (the MEASURED mean KL-to-base over
    scored steps — reported for EVERY method, anchored or not) and the
    property term's own metrics."""
    (eval_question_ids, eval_concept_ids, eval_responses,
     eval_n_valid_steps, eval_is_repeat) = eval_data
    scored_windows = collect_scored_windows(
        model, eval_question_ids, eval_concept_ids, eval_responses,
        eval_n_valid_steps, is_repeat=eval_is_repeat)
    labels, probs = flatten(scored_windows)
    metrics = dict(auc=auc(labels, probs), nll=nll(labels, probs),
                   fkl=mean_forward_kl(base_model, model, eval_data))
    metrics.update(property_term.eval_metrics(scored_windows))
    return metrics


def post_train(args, property_term):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    train_data, eval_data = load_train_eval(args.dataset)
    (train_question_ids, train_concept_ids, train_responses,
     train_n_valid_steps, _train_is_repeat) = train_data
    property_term.setup(*train_data)
    print(f"train {len(train_question_ids)}, eval {len(eval_data[0])}, "
          f"device={DEVICE}")

    model = build_model(args.model, args.dataset, args.ckpt)
    model.eval()      # eval mode but grads still flow (no dropout)
    if any(isinstance(m, torch.nn.RNNBase) for m in model.modules()):
        # cuDNN refuses an RNN backward in eval mode, and eval-mode-with-
        # grads IS the protocol here. Native kernels, same math.
        torch.backends.cudnn.enabled = False
        print("cudnn disabled: recurrent model post-trained in eval mode")
    if args.lora_rank:
        # LoRA adapters on a frozen base (bit-exact at init: B = 0)
        from train.lora import inject_lora
        inject_lora(model, args.lora_rank)
    # the frozen base: the anchor target (--beta) and the fkl measurement
    # reference for every method
    base_model = build_model(args.model, args.dataset, args.ckpt)
    for p in base_model.parameters():
        p.requires_grad_(False)
    # L2-SP: snapshot the INIT weights
    init_weights = None
    if args.l2sp:
        init_weights = {name: p.detach().clone()
                        for name, p in model.named_parameters()}
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)

    metrics_at_start = evaluate(model, eval_data, property_term, base_model)
    print(f"iter 0 baseline: {json.dumps(metrics_at_start)}")

    step_positions = torch.arange(T_MAX, device=DEVICE)[None, :]
    t_start = time.time()
    metrics = metrics_at_start
    for it in range(1, ITERS + 1):
        # sample a batch of windows; deduplicate so each window appears once
        sampled = torch.randint(len(train_question_ids), (BATCH,))
        in_batch_mask = torch.zeros(len(train_question_ids), dtype=torch.bool)
        in_batch_mask[sampled] = True
        window_indices = torch.nonzero(in_batch_mask).squeeze(1)
        batch_question_ids = train_question_ids[window_indices].to(DEVICE)
        batch_concept_ids = train_concept_ids[window_indices].to(DEVICE)
        batch_responses = train_responses[window_indices].to(DEVICE)
        batch_n_valid = train_n_valid_steps[window_indices].to(DEVICE)
        dcur = make_dcur(batch_question_ids, batch_concept_ids,
                         batch_responses)

        p_correct = forward_preds(model, dcur)
        # score positions 1..n_valid-1: real answers with at least one step
        # of history (position 0 has nothing to condition on)
        scored_steps_mask = ((step_positions >= 1)
                             & (step_positions < batch_n_valid[:, None]))
        p_correct_f64 = p_correct.clamp(1e-5, 1 - 1e-5).double()
        batch_ctx = dict(responses=batch_responses,
                         window_indices=window_indices,
                         in_batch_mask=in_batch_mask)

        property_value = property_term.value(p_correct, scored_steps_mask,
                                             batch_ctx)
        loss = args.lam * property_value
        if args.beta:
            with torch.no_grad():
                p_base = forward_preds(base_model, dcur) \
                    .clamp(1e-5, 1 - 1e-5).double()
            # closed-form per-step divergence to the frozen base
            loss = loss + args.beta * anchor_divergence(
                args.divergence, p_base, p_correct_f64
            )[scored_steps_mask].mean()
        if args.gamma:
            responses_f64 = batch_responses.double()
            loss = loss + args.gamma * -(
                responses_f64 * p_correct_f64.log()
                + (1 - responses_f64) * (1 - p_correct_f64).log()
            )[scored_steps_mask].mean()
        if init_weights is not None:
            penalty = torch.zeros((), dtype=torch.float64, device=DEVICE)
            for name, p in model.named_parameters():
                penalty = penalty + ((p - init_weights[name]) ** 2
                                     ).sum().double()
            loss = loss + args.l2sp / 2 * penalty
        optimizer.zero_grad()
        loss.backward()
        # grad clipping at 1.0 — stability guard
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % EVAL_EVERY == 0 or it == ITERS:
            metrics = evaluate(model, eval_data, property_term, base_model)
            per_it = (time.time() - t_start) / it
            print(f"iter {it:>5}  prop={float(property_value):.5f}  "
                  f"{json.dumps(metrics)}  [{per_it*1000:.0f} ms/iter]")

    torch.save(model.state_dict(), out)
    out.with_suffix(".final.json").write_text(json.dumps(
        {"args": vars(args), "baseline": metrics_at_start,
         "final": metrics}, indent=2))
    print(f"done in {(time.time()-t_start)/60:.1f} min -> {out}")


def run_cli(description, property_term_factory):
    """main() of a loss dir: parse args, build the term, adapt."""
    ap = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_DEFAULTS))
    ap.add_argument("--dataset", required=True, choices=sorted(DATASET_CFG))
    ap.add_argument("--ckpt", required=True,
                    help="the already-trained model: initialization AND "
                         "frozen anchor")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=1.0,
                    help="weight of the property term")
    ap.add_argument("--beta", type=float, default=0.0,
                    help="weight of the prediction-space anchor")
    ap.add_argument("--divergence", default="forward_kl",
                    choices=("forward_kl", "reverse_kl", "logit_mse"),
                    help="the anchor's divergence")
    ap.add_argument("--gamma", type=float, default=0.0,
                    help="weight of the BCE term")
    ap.add_argument("--l2sp", type=float, default=0.0,
                    help="weight of the L2-SP weight-space penalty")
    ap.add_argument("--lora-rank", type=int, default=0,
                    help="train LoRA adapters of this rank on a frozen base")
    args = ap.parse_args()
    post_train(args, property_term_factory(args))
