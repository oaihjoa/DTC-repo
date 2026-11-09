"""Equal-opportunity gap of a nips_task34 checkpoint (paper Table 3).

Metric source: equal opportunity, Hardt et al. NeurIPS 2016
(arXiv:1610.02413), in its threshold-free form over scored fold-4 rows:
  eo = | mean(p | r=1, A) - mean(p | r=1, B) |

    python evaluation/evaluation_eo.py --model qikt --ckpt ckpt/qikt_nips_eo.pt
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval_common import (MODEL_DEFAULTS, csv_path, load_windows,
                         build_model, collect_scored_windows)  # noqa: E402
from eo.loss import window_groups, eo_gap                      # noqa: E402

DATASET = "nips_task34"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_DEFAULTS))
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    (question_ids, concept_ids, responses, n_valid_steps, is_repeat,
     student_ids) = load_windows(csv_path(DATASET), [4], min_len=3,
                                 with_repeats=True, with_uid=True)
    model = build_model(args.model, DATASET, args.ckpt)
    scored_windows = collect_scored_windows(
        model, question_ids, concept_ids, responses, n_valid_steps,
        is_repeat=is_repeat)
    print(json.dumps({"eo": eo_gap(scored_windows,
                                   window_groups(student_ids))}))


if __name__ == "__main__":
    main()
