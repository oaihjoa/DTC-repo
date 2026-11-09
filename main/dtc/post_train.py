"""Post-hoc DTC optimization (the paper's headline arm).

    L = lam * (1 - DTC(p_theta)) + gamma*BCE + beta * mean KL(base||theta)
        (the constant 1 is dropped: the loop minimizes lam * (-DTC))
Paper settings: --loss prop+kl, lam 1, beta 64, lr 5e-6, 1500 iters.

Run: python dtc/post_train.py --loss prop+kl --lam 1 --beta 64 \
        --csv data/peiyou/train_valid_sequences.csv --dataset peiyou \
        --model qikt --ckpt ckpt/qikt_peiyou_base.pt \
        --out ckpt/qikt_peiyou_dtc_kl64.pt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.post_train_common import run_cli                  # noqa: E402
from dtc.loss import DTCTerm                                 # noqa: E402


if __name__ == "__main__":
    run_cli(__doc__, lambda args: DTCTerm())
