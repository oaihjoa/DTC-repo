"""Post-hoc DTC optimization (the paper's headline method).

    L = lam * (1 - DTC(p_theta)) + beta * mean KL(base || theta)
        (the constant 1 is dropped: the loop minimizes lam * (-DTC))

    python dtc/post_train.py --model qikt --dataset peiyou \
        --ckpt ckpt/qikt_peiyou.pt --beta 64 --out ckpt/qikt_peiyou_kl64.pt

The other preservation mechanisms: see train/post_train_common.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.post_train_common import run_cli                  # noqa: E402
from dtc.loss import DTCTerm                                 # noqa: E402


if __name__ == "__main__":
    run_cli(__doc__, lambda args: DTCTerm())
