"""Soft equal-opportunity + KL post-hoc adaptation (nips_task34 only — the
dataset with a demographic attribute). Stage 1 of the paper's Table 3.

    L = lam * |E[p|r=1,A]-E[p|r=1,B]| + gamma*BCE + beta * mean KL(base||theta)
Paper settings: --loss prop+kl, lam 25, beta 8, lr 5e-6, 1500 iters.
The audit of a checkpoint's EO gap is evaluation/evaluation_eo.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.post_train_common import run_cli                  # noqa: E402
from eo.loss import EqOppTerm                                # noqa: E402


def extra(ap):
    ap.add_argument("--groups",
                    default=str(Path(__file__).resolve().parents[1]
                                / "data/groups_nips_task34.json"))
    ap.add_argument("--group-attr", default="gender")


if __name__ == "__main__":
    run_cli(__doc__, EqOppTerm, extra)
