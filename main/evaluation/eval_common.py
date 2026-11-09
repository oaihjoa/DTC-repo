"""Shared data loading, model building and scoring.

NAMING CONVENTION (repo-wide): tensors are named for what they hold.
  question_ids   (n_windows, 200) long — pyKT question (problem) id per step
  concept_ids    (n_windows, 200) long — knowledge-concept id per step
  responses      (n_windows, 200) long — 1 correct / 0 wrong (padding -> 0)
  n_valid_steps  (n_windows,)     long — how many steps are real answers
                                         (the rest of the window is padding)
  is_repeat      (n_windows, 200) long — 1 on KC-expansion sibling rows
                                         (same question repeated once per
                                         extra concept it carries)

INPUT: pyKT's KC-expanded `train_valid_sequences.csv` (see README) and a
torch state_dict checkpoint.

OUTPUT: `collect_scored_windows` returns, per window, the triple
(concept_ids, responses, p_correct) over KEPT rows — the object every
metric consumes. Under the aligned protocol KC-expansion sibling rows are
predicted from but not scored. `auc`/`nll` implement:
  AUC via the rank statistic;  NLL = -mean[ r log p + (1-r) log(1-p) ].
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T_MAX = 200          # pyKT window length; positions beyond n_valid_steps are padding

DATASET_CFG = {  # n_question (distinct concepts), n_pid (distinct questions)
    "peiyou": dict(n_question=865, n_pid=7652),
    "assist2009": dict(n_question=123, n_pid=17737),
    "nips_task34": dict(n_question=57, n_pid=948),
}

MODEL_DEFAULTS = {
    "simplekt": dict(d_model=256, n_blocks=2, dropout=0.05, d_ff=256,
                     seq_len=T_MAX, kq_same=1, final_fc_dim=512,
                     final_fc_dim2=256, num_attn_heads=8, separate_qa=False,
                     emb_type="qid"),
    "sparsekt": dict(d_model=256, n_blocks=2, dropout=0.1, d_ff=256,
                     num_attn_heads=4, seq_len=T_MAX, sparse_ratio=0.8,
                     k_index=5, emb_type="qid_sparseattn"),
    # pyKT kt_config defaults
    "dkt": dict(emb_size=200, dropout=0.1),
    "dkvmn": dict(dim_s=200, size_m=50, dropout=0.2),
    "akt": dict(d_model=256, n_blocks=1, dropout=0.05, d_ff=256),
    "deep_irt": dict(dim_s=200, size_m=50, dropout=0.2),
    # official wandb_qikt_train.py defaults
    "qikt": dict(emb_size=300, dropout=0.4, mlp_layer_num=2),
    # official ukt_train.py defaults
    "ukt": dict(d_model=256, n_blocks=4, dropout=0.2, d_ff=512,
                num_attn_heads=8, seq_len=200),
}


def build_model(model_name, dataset, ckpt_path, cfg_json=""):
    """Construct the model and load a state_dict checkpoint."""
    from train.model_adapters import MODELS
    cfg = dict(MODEL_DEFAULTS[model_name], **DATASET_CFG[dataset])
    if cfg_json:
        cfg.update(json.loads(cfg_json))
    model = MODELS[model_name](**cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.to(DEVICE).eval()


def load_windows(csv_path, folds, min_len, with_repeats=False, with_uid=False):
    """Read pyKT windows into tensors.

    Keeps the windows of the given folds whose count of non-padding
    steps is >= min_len, in csv row order.
    Returns (question_ids, concept_ids, responses, n_valid_steps
             [, is_repeat] [, student_ids]).
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    if folds is not None:
        df = df[df["fold"].isin(folds)]
    all_questions, all_concepts, all_responses = [], [], []
    all_n_valid, all_is_repeat, all_student_ids = [], [], []
    for student_id, question_str, concept_str, response_str, repeat_str in zip(
            df["uid"], df["questions"], df["concepts"], df["responses"],
            df["is_repeat"]):
        response_row = [int(x) for x in response_str.split(",")]
        # padding cells hold -1; real answers are 0/1
        n_valid = sum(1 for x in response_row if x != -1)
        if n_valid < min_len:
            continue
        # clamp -1 padding to 0 so ids/responses are valid embedding inputs
        # (verified == pyKT's mask-multiply for these models)
        all_questions.append([max(int(x), 0) for x in question_str.split(",")])
        all_concepts.append([max(int(x), 0) for x in concept_str.split(",")])
        all_responses.append([max(x, 0) for x in response_row])
        all_is_repeat.append([max(int(x), 0) for x in repeat_str.split(",")])
        all_n_valid.append(n_valid)
        all_student_ids.append(int(student_id))
    out = [torch.tensor(all_questions), torch.tensor(all_concepts),
           torch.tensor(all_responses), torch.tensor(all_n_valid)]
    if with_repeats:
        out.append(torch.tensor(all_is_repeat))
    if with_uid:
        out.append(torch.tensor(all_student_ids))
    return tuple(out)


def make_dcur(question_ids, concept_ids, responses):
    """Split full windows into pyKT's unshifted/shifted input views."""
    return {"qseqs": question_ids[:, :-1], "cseqs": concept_ids[:, :-1],
            "rseqs": responses[:, :-1],
            "shft_qseqs": question_ids[:, 1:],
            "shft_cseqs": concept_ids[:, 1:],
            "shft_rseqs": responses[:, 1:]}


def forward_preds(model, dcur):
    """Model forward -> (batch, 200) P(correct); unwrap tuple returns."""
    out = model(dcur)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def collect_scored_windows(model, question_ids, concept_ids, responses,
                           n_valid_steps, is_repeat=None, bs=256):
    """Per window: (concept_ids, responses, p_correct) over KEPT rows.

    Predictions always consume the FULL history; only SCORING rows are
    dropped when is_repeat is given (the aligned protocol: KC-expansion
    sibling rows share their question's response and would leak).
    """
    scored_windows = []
    for lo in range(0, len(question_ids), bs):
        batch_q, batch_c, batch_r = (t[lo:lo + bs].to(DEVICE)
                                     for t in (question_ids, concept_ids,
                                               responses))
        p_correct = forward_preds(model,
                                  make_dcur(batch_q, batch_c, batch_r))
        p_correct = p_correct.cpu().numpy()
        for i in range(len(batch_q)):
            n_valid = int(n_valid_steps[lo + i])
            # keep every real step, or only non-repeat rows (aligned)
            keep = (slice(None) if is_repeat is None
                    else np.flatnonzero(is_repeat[lo + i, :n_valid].numpy() == 0))
            scored_windows.append((
                concept_ids[lo + i, :n_valid].numpy()[keep].tolist(),
                responses[lo + i, :n_valid].numpy()[keep].tolist(),
                np.clip(p_correct[i, :n_valid][keep], 1e-6, 1 - 1e-6).tolist()))
    return scored_windows


def flatten(scored_windows):
    """Concatenate all kept rows -> (labels, probabilities) arrays."""
    labels = np.array([r for _, window_responses, _ in scored_windows
                       for r in window_responses], dtype=float)
    probs = np.array([p for _, _, window_probs in scored_windows
                      for p in window_probs])
    return labels, probs


def auc(labels, probs):
    """Rank-statistic AUC: P(random correct scores above random wrong)."""
    order = probs.argsort(kind="mergesort")
    rank = np.empty(len(probs)); rank[order] = np.arange(1, len(probs) + 1)
    n_correct = labels.sum()
    if n_correct == 0 or n_correct == len(labels):
        return float("nan")     # AUC undefined with a single class
    return float((rank[labels == 1].sum() - n_correct * (n_correct + 1) / 2)
                 / (n_correct * (len(labels) - n_correct)))


def nll(labels, probs):
    """Mean negative log-likelihood (logloss) of the kept rows."""
    return float(-np.mean(labels * np.log(probs)
                          + (1 - labels) * np.log(1 - probs)))

