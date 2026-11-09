"""Base-model training (next-response BCE), the paper's recipe (Sec. 5.3).

    python train/train_base.py --model qikt --dataset peiyou --out ckpt/qikt_peiyou.pt

INPUT   data/<dataset>/train_valid_sequences.csv (pyKT's KC-expanded split),
        train folds {0,1,2,3}, valid fold 4.
OUTPUT  best-valid-AUC state_dict -> <out>
        + <out stem>.train.json (best AUC/epoch, minutes).

OBJECTIVE
    L = -mean_{(b,t): 1 <= t < n_valid_steps_b}
            [ r_bt log p_bt + (1-r_bt) log(1-p_bt) ]
        + the model's own published auxiliary loss, when it has one
          (AKT's Rasch L2, UKT's contrastive term)
RECIPE  Adam at the model's learning rate, halved on a valid-AUC plateau;
        an EMA of the weights is what gets validated, selected and saved;
        selection uses the same aligned protocol as the post-hoc
        evaluation; early stop after 20 epochs without improvement.
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
                         csv_path, load_windows, make_dcur, forward_preds,
                         collect_scored_windows, flatten, auc)  # noqa: E402
from train.model_adapters import MODELS                        # noqa: E402

LEARNING_RATE = {"dkt": 1e-3, "dkvmn": 1e-3, "deep_irt": 1e-3, "qikt": 1e-3,
                 "simplekt": 1e-4, "sparsekt": 1e-4, "ukt": 1e-4,
                 "akt": 1e-5}
BATCH_SIZE = {"ukt": 128}     # UKT's contrastive term doubles its memory
DEFAULT_BATCH_SIZE = 256
MAX_EPOCHS = 500
PATIENCE = 20                 # epochs without valid-AUC improvement
EMA_HORIZON_EPOCHS = 3        # EMA decay = 1 - 1/(horizon * steps_per_epoch)


def train_bce(model, model_name, dataset, out):
    csv = csv_path(dataset)
    (train_question_ids, train_concept_ids, train_responses,
     train_n_valid_steps) = load_windows(csv, [0, 1, 2, 3], min_len=8)
    (eval_question_ids, eval_concept_ids, eval_responses,
     eval_n_valid_steps, eval_is_repeat) = load_windows(
        csv, [4], min_len=3, with_repeats=True)
    print(f"train {len(train_question_ids)} windows, "
          f"valid {len(eval_question_ids)}, device={DEVICE}")

    batch_size = BATCH_SIZE.get(model_name, DEFAULT_BATCH_SIZE)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE[model_name])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)
    steps_per_epoch = max(1, (len(train_question_ids) + batch_size - 1)
                          // batch_size)
    ema_decay = 1.0 - 1.0 / (EMA_HORIZON_EPOCHS * steps_per_epoch)
    ema_weights = {k: v.detach().clone()
                   for k, v in model.state_dict().items()}
    # (1, T_MAX) row of step positions; broadcast against each window's
    # n_valid_steps below to build the per-step scoring mask
    step_positions = torch.arange(T_MAX, device=DEVICE)[None, :]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    best_auc, best_epoch = 0.0, -1
    t_start = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        # fresh random window order every epoch (pyKT's per-epoch shuffle)
        order = torch.randperm(len(train_question_ids))
        losses = []
        for lo in range(0, len(order), batch_size):
            j = order[lo:lo + batch_size]
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
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema_weights[k].mul_(ema_decay).add_(
                            v, alpha=1 - ema_decay)
                    else:
                        ema_weights[k].copy_(v)
            losses.append(loss.item())

        # validate/select/save on the AVERAGED weights: swap in the EMA,
        # restore the raw iterate after the epoch's bookkeeping
        model.eval()
        raw_weights = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}
        model.load_state_dict(ema_weights)
        labels, probs = flatten(collect_scored_windows(
            model, eval_question_ids, eval_concept_ids, eval_responses,
            eval_n_valid_steps, is_repeat=eval_is_repeat))
        valid_auc = auc(labels, probs)
        marker = ""
        if valid_auc > best_auc:
            best_auc, best_epoch = valid_auc, epoch
            torch.save(model.state_dict(), out)
            marker = "  *saved"
        print(f"epoch {epoch:>3}  loss={np.mean(losses):.4f}  "
              f"valid_auc={valid_auc:.4f}{marker}")
        model.load_state_dict(raw_weights)
        scheduler.step(valid_auc)
        if epoch - best_epoch >= PATIENCE:
            print(f"early stop (patience {PATIENCE})")
            break

    out.with_suffix(".train.json").write_text(json.dumps(
        {"model": model_name, "dataset": dataset,
         "best_valid_auc": best_auc, "best_epoch": best_epoch,
         "minutes": (time.time() - t_start) / 60}, indent=2))
    print(f"best valid AUC {best_auc:.4f} @ epoch {best_epoch} -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--dataset", required=True, choices=sorted(DATASET_CFG))
    ap.add_argument("--out", required=True, help="checkpoint path")
    args = ap.parse_args()
    # model hyperparameters + dataset sizes (n_question / n_pid)
    cfg = dict(MODEL_DEFAULTS[args.model], **DATASET_CFG[args.dataset])
    train_bce(MODELS[args.model](**cfg), args.model, args.dataset, args.out)


if __name__ == "__main__":
    main()
