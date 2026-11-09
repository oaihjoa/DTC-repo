"""The post-hoc adaptation loop shared by the property directories.

THE OBJECTIVE (paper Sec. 4.4):

    L(theta) = lam * Property(p_theta)                       [the loss dir's term]
             + gamma * BCE(p_theta, responses)               [optional]
             + beta  * mean_{scored steps} D( p_base , p_theta )   [optional]

    D defaults to the forward Bernoulli KL, frozen base first:
        KL(Bern(a)||Bern(b)) = a log(a/b) + (1-a) log((1-a)/(1-b))
    closed form per step, no sampling.

PRESERVATION MECHANISMS (paper Sec. 5.4), all on the same loop:
    no anchor   --loss prop
    KL (ours)   --loss prop+kl                             (knob: --beta)
    reverse KL  --loss prop+kl --divergence reverse_kl     (knob: --beta)
    logit-MSE   --loss prop+kl --divergence logit_mse      (knob: --beta)
    BCE         --loss prop+bce                            (knob: --gamma)
    L2-SP       --loss prop --weight-anchor l2sp           (knob: --weight-alpha)
    LoRA        --loss prop+bce --train-scope lora         (knob: --lora-rank)
FRAPPE and WiSE-FT live in baselines/ (they are not trained by this loop).

TRAINING PROTOCOL: model.eval() with grads flowing (no dropout),
Adam(lr 5e-6 default), grad-clip 1.0, batch = unique of 256 sampled
windows, 1500 iters, eval every 250 on the aligned fold-4 protocol.

INTERFACE for a loss dir:  post_train(args, property_term) where
property_term is an object with
    .setup(train_question_ids, train_concept_ids, train_responses,
           train_n_valid_steps, train_is_repeat)  -> None
    .value(p_correct, scored_steps_mask, batch_ctx) -> DIFFERENTIABLE scalar
    .eval_metrics(scored_windows, model, eval_data) -> dict of floats
        (eval_data = the five eval tensors; most terms only use the windows)
batch_ctx keys: question_ids, concept_ids, responses, n_valid_steps
    (the batch tensors), window_indices (which train windows are in the
    batch, sorted), in_batch_mask (bool over ALL train windows), model.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from eval_common import (DEVICE, T_MAX, MODEL_DEFAULTS,
                         load_windows, make_dcur, forward_preds, build_model,
                         collect_scored_windows, flatten, auc, nll)  # noqa: E402


def add_post_train_args(ap):
    ap.add_argument("--csv", required=True)
    ap.add_argument("--dataset", default="peiyou")
    ap.add_argument("--model", default="simplekt",
                    choices=tuple(MODEL_DEFAULTS))
    ap.add_argument("--model-cfg", default="")
    ap.add_argument("--ckpt", required=True,
                    help="the already-trained model: initialization AND "
                         "frozen anchor")
    ap.add_argument("--out", required=True)
    ap.add_argument("--loss", required=True,
                    help="'+'-joined subset of {prop,bce,kl}, e.g. prop+kl")
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=8.0)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=0)
    ap.add_argument("--divergence", default="forward_kl",
                    choices=("forward_kl", "reverse_kl", "logit_mse"),
                    help="anchor divergence of the 'kl' loss component: "
                         "forward Bernoulli KL (default/ours), reverse KL, "
                         "or logit MSE (function-space L2)")
    ap.add_argument("--train-scope", default="all",
                    choices=("all", "lora"),
                    help="which parameters train: all (default) or LoRA "
                         "adapters on a frozen base")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--weight-anchor", default="none",
                    choices=("none", "l2sp"),
                    help="l2sp: weight-space L2 penalty to the INIT weights "
                         "(Li et al., ICML 2018)")
    ap.add_argument("--weight-alpha", type=float, default=15.0)
    return ap


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
            # closed-form forward Bernoulli KL, summed over scored steps
            fkl_sum += float((p_ref * (p_ref / p_now).log()
                              + (1 - p_ref) * ((1 - p_ref)
                                               / (1 - p_now)).log()
                              )[mask].sum())
            n_steps += int(mask.sum())
    return fkl_sum / max(n_steps, 1)


def evaluate(model, eval_data, property_term, base_model=None):
    """Aligned fold-4 evaluation: accuracy columns + the arm's own metrics
    + (when the frozen base is supplied) fkl, the MEASURED mean KL-to-base
    over scored steps — logged for EVERY arm, anchored or not, so arms can
    be compared by their achieved functional drift."""
    (eval_question_ids, eval_concept_ids, eval_responses,
     eval_n_valid_steps, eval_is_repeat) = eval_data
    scored_windows = collect_scored_windows(
        model, eval_question_ids, eval_concept_ids, eval_responses,
        eval_n_valid_steps, is_repeat=eval_is_repeat)
    labels, probs = flatten(scored_windows)
    metrics = dict(auc=auc(labels, probs), nll=nll(labels, probs))
    if base_model is not None:
        metrics["fkl"] = mean_forward_kl(base_model, model, eval_data)
    metrics.update(property_term.eval_metrics(scored_windows, model,
                                              eval_data))
    return metrics


def post_train(args, property_term):
    loss_components = args.loss.split("+")
    assert loss_components and set(loss_components) <= {"prop", "bce", "kl"} \
        and len(set(loss_components)) == len(loss_components), \
        f"bad --loss {args.loss!r}"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # train folds 0-3, eval fold 4
    (train_question_ids, train_concept_ids, train_responses,
     train_n_valid_steps, train_is_repeat) = load_windows(
        args.csv, [0, 1, 2, 3], min_len=8, with_repeats=True)
    (eval_question_ids, eval_concept_ids, eval_responses,
     eval_n_valid_steps, eval_is_repeat) = load_windows(
        args.csv, [4], min_len=3, with_repeats=True)
    if args.n_train:      # smoke-test caps
        (train_question_ids, train_concept_ids, train_responses,
         train_n_valid_steps, train_is_repeat) = (
            t[:args.n_train] for t in
            (train_question_ids, train_concept_ids, train_responses,
             train_n_valid_steps, train_is_repeat))
    if args.n_eval:
        (eval_question_ids, eval_concept_ids, eval_responses,
         eval_n_valid_steps) = (t[:args.n_eval] for t in
                                (eval_question_ids, eval_concept_ids,
                                 eval_responses, eval_n_valid_steps))
        eval_is_repeat = eval_is_repeat[:args.n_eval]
    eval_data = (eval_question_ids, eval_concept_ids, eval_responses,
                 eval_n_valid_steps, eval_is_repeat)
    property_term.setup(train_question_ids, train_concept_ids,
                        train_responses, train_n_valid_steps, train_is_repeat)
    print(f"loss={args.loss}  train {len(train_question_ids)}, "
          f"eval {len(eval_question_ids)}, device={DEVICE}")

    model = build_model(args.model, args.dataset, args.ckpt, args.model_cfg)
    model.eval()      # eval mode but grads still flow (no dropout)
    if any(isinstance(m, torch.nn.RNNBase) for m in model.modules()):
        # cuDNN refuses an RNN backward in eval mode, and eval-mode-with-
        # grads IS the protocol here. Native kernels, same math.
        torch.backends.cudnn.enabled = False
        print("cudnn disabled: recurrent model post-trained in eval mode")
    if args.train_scope == "lora":
        # LoRA adapters on a frozen base (bit-exact at init: B = 0)
        from train.lora import inject_lora
        inject_lora(model, args.lora_rank)
    # the frozen base is ALWAYS built: it is the anchor target when 'kl' is
    # a loss component, and the fkl measurement reference for every arm
    base_model = build_model(args.model, args.dataset, args.ckpt,
                             args.model_cfg)
    for p in base_model.parameters():
        p.requires_grad_(False)
    # L2-SP: snapshot the INIT weights
    init_weights = None
    if args.weight_anchor == "l2sp":
        init_weights = {name: p.detach().clone()
                        for name, p in model.named_parameters()}
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    metrics_at_start = evaluate(model, eval_data, property_term,
                                base_model=base_model)
    print(f"iter 0 baseline: {json.dumps(metrics_at_start)}")
    log_file = open(out.with_suffix(".metrics.jsonl"), "w")
    log_file.write(json.dumps({"iter": 0, **metrics_at_start}) + "\n")
    log_file.flush()

    step_positions = torch.arange(T_MAX, device=DEVICE)[None, :]
    t_start = time.time()
    metrics = metrics_at_start
    for it in range(1, args.iters + 1):
        # sample a batch of windows; deduplicate so each window appears once
        sampled = torch.randint(len(train_question_ids), (args.batch,))
        in_batch_mask = torch.zeros(len(train_question_ids), dtype=torch.bool)
        in_batch_mask[sampled] = True
        window_indices = torch.nonzero(in_batch_mask).squeeze(1)
        batch_question_ids = train_question_ids[window_indices].to(DEVICE)
        batch_concept_ids = train_concept_ids[window_indices].to(DEVICE)
        batch_responses = train_responses[window_indices].to(DEVICE)
        batch_n_valid = train_n_valid_steps[window_indices].to(DEVICE)

        p_correct = forward_preds(model, make_dcur(
            batch_question_ids, batch_concept_ids, batch_responses))
        # score positions 1..n_valid-1: real answers with at least one step
        # of history (position 0 has nothing to condition on)
        scored_steps_mask = ((step_positions >= 1)
                             & (step_positions < batch_n_valid[:, None]))
        p_correct_f64 = p_correct.clamp(1e-5, 1 - 1e-5).double()
        responses_f64 = batch_responses.double()
        batch_ctx = dict(question_ids=batch_question_ids,
                         concept_ids=batch_concept_ids,
                         responses=batch_responses,
                         n_valid_steps=batch_n_valid,
                         window_indices=window_indices,
                         in_batch_mask=in_batch_mask, model=model)

        property_value = property_term.value(p_correct, scored_steps_mask,
                                             batch_ctx)
        # standard logloss on the scored steps (optional component)
        bce = -(responses_f64 * p_correct_f64.log()
                + (1 - responses_f64)
                * (1 - p_correct_f64).log())[scored_steps_mask].mean()
        if "kl" in loss_components:
            with torch.no_grad():
                p_base = forward_preds(base_model, make_dcur(
                    batch_question_ids, batch_concept_ids, batch_responses))
                p_base = p_base.clamp(1e-5, 1 - 1e-5).double()
            # closed-form per-step divergence to the frozen base
            kl = anchor_divergence(args.divergence, p_base, p_correct_f64
                                   )[scored_steps_mask].mean()
        else:
            kl = torch.zeros((), dtype=torch.float64, device=DEVICE)

        # assemble only the requested components
        loss = torch.zeros((), dtype=torch.float64, device=DEVICE)
        if "prop" in loss_components:
            loss = loss + args.lam * property_value
        if "bce" in loss_components:
            loss = loss + args.gamma * bce
        if "kl" in loss_components:
            loss = loss + args.beta * kl
        if init_weights is not None:
            penalty = torch.zeros((), dtype=torch.float64, device=DEVICE)
            for name, p in model.named_parameters():
                penalty = penalty + ((p - init_weights[name]) ** 2
                                     ).sum().double()
            loss = loss + args.weight_alpha / 2 * penalty
        optimizer.zero_grad()
        loss.backward()
        # grad clipping at 1.0 — stability guard
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % args.eval_every == 0 or it == args.iters:
            metrics = evaluate(model, eval_data, property_term,
                               base_model=base_model)
            per_it = (time.time() - t_start) / it
            print(f"iter {it:>5}  prop={float(property_value):.5f} "
                  f"bce={bce.item():.4f} kl={kl.item():.5f}  "
                  f"{json.dumps(metrics)}  [{per_it*1000:.0f} ms/iter]")
            log_file.write(json.dumps({"iter": it,
                                       "prop": float(property_value),
                                       "bce": bce.item(), "kl": kl.item(),
                                       **metrics}) + "\n")
            log_file.flush()

    torch.save(model.state_dict(), out)
    out.with_suffix(".final.json").write_text(json.dumps(
        {"loss": args.loss, "args": vars(args),
         "baseline": metrics_at_start, "final": metrics}, indent=2))
    print(f"done in {(time.time()-t_start)/60:.1f} min -> {out}")


def run_cli(description, property_term_factory, extra_args=None):
    """Standard main() for a loss dir: parse args, build the term, train."""
    import argparse
    ap = add_post_train_args(argparse.ArgumentParser(description=description))
    if extra_args:
        extra_args(ap)
    args = ap.parse_args()
    post_train(args, property_term_factory(args))
