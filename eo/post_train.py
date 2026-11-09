"""Soft equal-opportunity + KL post-hoc adaptation (nips_task34 only — the
dataset with a demographic attribute). Stage 1 of the paper's Table 3.

    L = lam * |E[p|r=1,A] - E[p|r=1,B]| + beta * mean KL(base || theta)

    python eo/post_train.py --model qikt --dataset nips_task34 \
        --ckpt ckpt/qikt_nips.pt --lam 25 --beta 8 --out ckpt/qikt_nips_eo.pt

Needs data/groups_nips_task34.json (eo/join_metadata.py). The EO gap of any
checkpoint: evaluation/evaluation_eo.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.post_train_common import run_cli                  # noqa: E402
from eo.loss import EqOppTerm                                # noqa: E402


if __name__ == "__main__":
    run_cli(__doc__, EqOppTerm)
