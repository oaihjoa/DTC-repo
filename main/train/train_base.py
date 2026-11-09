"""Base-model training (next-response BCE) for all eight models.

INPUT   pyKT KC-expanded csv (data/<dataset>/train_valid_sequences.csv),
        train folds {0,1,2,3}, valid fold 4.
OUTPUT  best-valid-AUC state_dict -> <out>.pt
        + <out>.train.json (best AUC/epoch, args, minutes).

OBJECTIVE (mirrors pyKT's official training):
    L = -mean_{(b,t): 1 <= t < n_valid_steps_b}
            [ r_bt log p_bt + (1-r_bt) log(1-p_bt) ]
        + the model's own published auxiliary loss, when it has one
          (AKT's Rasch L2; UKT's contrastive term with
          --model-cfg '{"use_cl": 1}')
    Adam(lr), no grad-clip (pyKT), batch shuffle per epoch, early stop on
    valid AUC.
Flag defaults = pyKT kt_config (batch 256, lr 1e-4, epochs 200, patience
10, min-delta 1e-3, seed 42). The paper's base recipe overrides them:
    --epochs 500 --patience 20 --min-delta 0 --lr-schedule plateau
    --valid-protocol aligned --ema-horizon-epochs 3   (+ per-model --lr)
see experiments/common.sh.

Run: python train/train_base.py --model akt \
        --csv data/peiyou/train_valid_sequences.csv --dataset peiyou \
        --out ckpt/akt_peiyou_base.pt
Smoke: append  --n-seqs 200 --epochs 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from eval_common import (DEVICE, T_MAX, DATASET_CFG, MODEL_DEFAULTS,
                         load_windows, make_dcur, forward_preds,
                         collect_scored_windows, flatten, auc)  # noqa: E402
from train.model_adapters import MODELS                        # noqa: E402


def add_args(ap):
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--csv", required=True)
    ap.add_argument("--dataset", default="peiyou")
    ap.add_argument("--model-cfg", default="", help="JSON overrides")
    ap.add_argument("--out", required=True, help="checkpoint path")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min-delta", type=float, default=1e-3,
                    help="early-stop improvement threshold. 1e-3 is the "
                         "pyKT convention; the paper's bases use 0 — any "
                         "improvement counts")
    ap.add_argument("--lr-schedule", default="none",
                    choices=("none", "plateau"),
                    help="none = pyKT's constant lr (default). plateau = "
                         "ReduceLROnPlateau on valid AUC (factor 0.5, "
                         "patience 5, min_lr 1e-6): the final checkpoint "
                         "sits at a small lr, so post-hoc adaptation has "
                         "no leftover BCE improvement to harvest")
    ap.add_argument("--valid-protocol", default="allsteps",
                    choices=("allsteps", "aligned"),
                    help="checkpoint-selection metric: allsteps = pyKT's "
                         "pretraining convention (every real step scores; "
                         "default). aligned = the post-hoc eval protocol "
                         "(repeat rows excluded) — selection and "
                         "downstream evaluation then agree")
    ap.add_argument("--ema-horizon-epochs", type=float, default=0.0,
                    help="if >0: validate, select and save an EMA of the "
                         "weights with decay = 1 - 1/(H * steps_per_epoch) "
                         "— the averaged basin center instead of the "
                         "trajectory's luckiest point. The paper uses H = 3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-seqs", type=int, default=0, help="cap train (smoke)")
    return ap


def train_bce(model, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    (train_question_ids, train_concept_ids, train_responses,
     train_n_valid_steps) = load_windows(args.csv, [0, 1, 2, 3], min_len=8)
    # eval_is_repeat is only used by --valid-protocol aligned; the default
    # validation AUC scores ALL real steps (pyKT's pretraining convention)
    (eval_question_ids, eval_concept_ids, eval_responses,
     eval_n_valid_steps, eval_is_repeat) = load_windows(
        args.csv, [4], min_len=3, with_repeats=True)
    if args.n_seqs:
        (train_question_ids, train_concept_ids, train_responses,
         train_n_valid_steps) = (
            t[:args.n_seqs] for t in
            (train_question_ids, train_concept_ids, train_responses,
             train_n_valid_steps))
    print(f"train {len(train_question_ids)} windows, "
          f"valid {len(eval_question_ids)}, device={DEVICE}")

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema_weights = None
    ema_decay = 0.0
    if args.ema_horizon_epochs > 0:
        steps_per_epoch = max(1, (len(train_question_ids)
                                  + args.batch_size - 1) // args.batch_size)
        ema_decay = 1.0 - 1.0 / (args.ema_horizon_epochs * steps_per_epoch)
        print(f"ema decay {ema_decay:.6f} from horizon "
              f"{args.ema_horizon_epochs} epochs x {steps_per_epoch} steps")
        ema_weights = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}
    scheduler = None
    if args.lr_schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)
    # (1, T_MAX) row of step positions; broadcast against each window's
    # n_valid_steps below to build the per-step scoring mask
    step_positions = torch.arange(T_MAX, device=DEVICE)[None, :]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    best_auc, best_epoch = 0.0, -1
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        # fresh random window order every epoch (pyKT's per-epoch shuffle)
        order = torch.randperm(len(train_question_ids))
        losses = []
        for lo in range(0, len(order), args.batch_size):
            j = order[lo:lo + args.batch_size]
            batch_question_ids, batch_concept_ids, batch_responses = (
                t[j].to(DEVICE) for t in
                (train_question_ids, train_concept_ids, train_responses))
            batch_n_valid_steps = train_n_valid_steps[j].to(DEVICE)
            p_correct = forward_preds(model, make_dcur(
                batch_question_ids, batch_concept_ids,
                batch_responses))                            # (B, T_MAX)
            # clamp away exact 0/1 before taking logs; float64 for stability
            p_correct_f64 = p_correct.clamp(1e-5, 1 - 1e-5).double()
            responses_f64 = batch_responses.double()
            # score positions 1..n_valid-1: real answers with at least one
            # step of history (position 0 has nothing to condition on;
            # positions >= n_valid_steps are padding)
            scored_steps_mask = (
                (step_positions >= 1)
                & (step_positions < batch_n_valid_steps[:, None]))
            # plain BCE averaged over the scored steps only
            loss = -(responses_f64 * p_correct_f64.log()
                     + (1 - responses_f64)
                     * (1 - p_correct_f64).log())[scored_steps_mask].mean()
            # models whose published recipe carries an auxiliary loss
            # (AKT's Rasch L2, UKT's contrastive term) expose it as
            # .last_reg_loss after forward
            reg = getattr(model, "last_reg_loss", None)
            if torch.is_tensor(reg):
                loss = loss + reg
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if ema_weights is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema_weights[k].mul_(ema_decay).add_(
                                v, alpha=1 - ema_decay)
                        else:
                            ema_weights[k].copy_(v)
            losses.append(loss.item())

        model.eval()
        if ema_weights is not None:
            # validate/select/save on the AVERAGED weights: swap in the
            # EMA, restore the raw iterate after the epoch's bookkeeping
            raw_weights = {k: v.detach().clone()
                           for k, v in model.state_dict().items()}
            model.load_state_dict(ema_weights)
        selection_repeats = (eval_is_repeat
                             if args.valid_protocol == "aligned" else None)
        labels, probs = flatten(collect_scored_windows(
            model, eval_question_ids, eval_concept_ids, eval_responses,
            eval_n_valid_steps, is_repeat=selection_repeats))
        valid_auc = auc(labels, probs)
        marker = ""
        if valid_auc > best_auc + args.min_delta:  # see --min-delta help
            best_auc, best_epoch = valid_auc, epoch
            torch.save(model.state_dict(), out)
            marker = "  *saved"
        print(f"epoch {epoch:>3}  loss={np.mean(losses):.4f}  "
              f"valid_auc={valid_auc:.4f}{marker}")
        if ema_weights is not None:
            model.load_state_dict(raw_weights)
        if scheduler is not None:
            scheduler.step(valid_auc)
        if epoch - best_epoch >= args.patience:
            print(f"early stop (patience {args.patience})")
            break

    out.with_suffix(".train.json").write_text(json.dumps(
        {"best_valid_auc": best_auc, "best_epoch": best_epoch,
         "args": vars(args), "minutes": (time.time() - t_start) / 60},
        indent=2))
    print(f"best valid AUC {best_auc:.4f} @ epoch {best_epoch} -> {out}")


def main():
    args = add_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    # model hyperparameters + dataset sizes (n_question/n_pid); an optional
    # --model-cfg JSON overrides either
    cfg = dict(MODEL_DEFAULTS[args.model], **DATASET_CFG[args.dataset])
    if args.model_cfg:
        cfg.update(json.loads(args.model_cfg))
    train_bce(MODELS[args.model](**cfg), args)


if __name__ == "__main__":
    main()
