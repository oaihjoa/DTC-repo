"""WiSE-FT baseline (eval-only; no training).

SOURCE: Wortsman et al., "Robust fine-tuning of zero-shot models",
CVPR 2022 (arXiv:2109.01903):
    theta(a) = theta_base + a * (theta_post - theta_base)
one evaluated model per interpolation coefficient a. In the paper
theta_post is the no-anchor DTC fine-tune (dtc/post_train.py with no
preservation flag). Each interpolated model is scored on fold 4 with fkl
measured against the base — the same (STCR, AUC, NLL, fkl) axes as every
trained method.

    python baselines/wise_ft.py --model qikt --dataset peiyou \
        --base ckpt/qikt_peiyou.pt --post ckpt/qikt_peiyou_noanchor.pt \
        --out ckpt/qikt_peiyou_wise.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from eval_common import (MODEL_DEFAULTS, DATASET_CFG, csv_path,
                         build_model, load_windows)           # noqa: E402
from train.post_train_common import evaluate                  # noqa: E402
from dtc.loss import DTCTerm                                  # noqa: E402

WISE_ALPHAS = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_DEFAULTS))
    ap.add_argument("--dataset", required=True, choices=sorted(DATASET_CFG))
    ap.add_argument("--base", required=True, help="base checkpoint")
    ap.add_argument("--post", required=True, help="fine-tuned checkpoint")
    ap.add_argument("--out", required=True, help="json of per-alpha rows")
    args = ap.parse_args()

    base_model = build_model(args.model, args.dataset, args.base)
    model = build_model(args.model, args.dataset, args.post)
    base_sd = {k: v.detach().clone()
               for k, v in base_model.state_dict().items()}
    post_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    delta = {k: post_sd[k] - base_sd[k]
             if post_sd[k].is_floating_point() else torch.zeros_like(
                 post_sd[k]) for k in post_sd}

    # min_len 2 (not the loop's 3): the setting the paper's WiSE-FT rows
    # were produced with
    eval_data = load_windows(csv_path(args.dataset), [4], min_len=2,
                             with_repeats=True)
    term = DTCTerm()

    rows = {"post_endpoint": evaluate(model, eval_data, term, base_model)}
    for alpha in WISE_ALPHAS:
        model.load_state_dict({k: base_sd[k] + alpha * d
                               for k, d in delta.items()})
        model.eval()
        rows[f"wise{alpha}"] = evaluate(model, eval_data, term, base_model)
        print(f"wise {alpha}: " + json.dumps(rows[f"wise{alpha}"]),
              flush=True)
    Path(args.out).write_text(json.dumps(
        dict(base=args.base, post=args.post, rows=rows), indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
