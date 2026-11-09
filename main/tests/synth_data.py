"""Synthetic pyKT-format data for the self-contained test suite.

Generates tests/synth/train_valid_sequences.csv with the exact column
contract of pyKT's KC-expanded file (uid, questions, concepts, responses,
is_repeat, fold; comma-joined, -1 padded to T=200) at toy scale, with
question GROUPS (constant question id + response within each is_repeat run)
as pyKT's KC-expansion produces them. Also writes a synthetic groups json
for the eo arm and tiny simplekt / qikt base checkpoints.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

T = 200
N_KC, N_PID = 20, 40
TINY_SIMPLEKT = dict(n_question=N_KC, n_pid=N_PID, d_model=16, n_blocks=1,
                     dropout=0.05, d_ff=16, seq_len=T, kq_same=1,
                     final_fc_dim=32, final_fc_dim2=16, num_attn_heads=2,
                     separate_qa=False, emb_type="qid")
TINY_QIKT = dict(n_question=N_KC, n_pid=N_PID, emb_size=16, dropout=0.4,
                 mlp_layer_num=2)
TINY_SPARSEKT = dict(n_question=N_KC, n_pid=N_PID, d_model=16, n_blocks=1,
                     dropout=0.1, d_ff=16, num_attn_heads=2, seq_len=T,
                     sparse_ratio=0.8, k_index=2, emb_type="qid_sparseattn")


def make_csv(path, n_windows=60, seed=0):
    rng = np.random.default_rng(seed)
    # fixed question -> KC map (1-3 KCs per question); drives the
    # KC-expansion below, mirroring pyKT's multi-concept questions
    q_to_kcs = {qid: rng.integers(0, N_KC, size=rng.integers(1, 4)).tolist()
                for qid in range(N_PID)}
    csv_rows = []
    for w in range(n_windows):
        uid = w // 2   # two windows per synthetic student
        n_valid_steps = int(rng.integers(30, 90))
        question_ids, concept_ids, responses, is_repeat = [], [], [], []
        while len(question_ids) < n_valid_steps:
            qid = int(rng.integers(0, N_PID))
            response = int(rng.random() < 0.7)
            # KC-expansion: emit one row per KC of the question, all sharing
            # the question id and response; rows after the first carry
            # is_repeat=1 (pyKT's convention), so each "question group" is a
            # constant-(qid, response) run
            for j, kc in enumerate(q_to_kcs[qid]):
                if len(question_ids) >= n_valid_steps:
                    break
                question_ids.append(qid)
                concept_ids.append(int(kc))
                responses.append(response)
                is_repeat.append(0 if j == 0 else 1)
        # pad every column out to T with pyKT's -1 sentinel, then comma-join
        pad = ["-1"] * (T - n_valid_steps)
        join_padded = lambda xs: ",".join([str(x) for x in xs] + pad)
        csv_rows.append({
            "fold": w % 5, "uid": uid,
            "questions": join_padded(question_ids),
            "concepts": join_padded(concept_ids),
            "responses": join_padded(responses),
            "selectmasks": join_padded(["1"] * n_valid_steps),
            "is_repeat": join_padded(is_repeat),
        })
    import pandas as pd
    pd.DataFrame(csv_rows).to_csv(path, index=False)
    return path


def make_groups_json(path, n_uids=30):
    # one binary demographic attribute over uids, deliberately using
    # non-{0,1} raw labels to exercise the eo relabeling
    gender = {u: (1 if u % 2 == 0 else 2) for u in range(n_uids)}
    Path(path).write_text(json.dumps({"gender": gender}))
    return path


def make_base_ckpt(path, model_name="simplekt", cfg=TINY_SIMPLEKT):
    # an untrained tiny model: enough for forward/backward smokes
    from train.model_adapters import MODELS
    torch.manual_seed(7)
    torch.save(MODELS[model_name](**cfg).state_dict(), path)
    return path


def ensure_all():
    synth_dir = ROOT / "tests" / "synth"
    synth_dir.mkdir(parents=True, exist_ok=True)
    csv = synth_dir / "train_valid_sequences.csv"
    if not csv.exists():
        make_csv(csv)
    groups = synth_dir / "groups.json"
    if not groups.exists():
        make_groups_json(groups)
    ckpt = synth_dir / "tiny_simplekt_base.pt"
    if not ckpt.exists():
        make_base_ckpt(ckpt)
    return csv, groups, ckpt


def ensure_qikt_base():
    ckpt = ROOT / "tests" / "synth" / "tiny_qikt_base.pt"
    if not ckpt.exists():
        make_base_ckpt(ckpt, "qikt", TINY_QIKT)
    return ckpt


if __name__ == "__main__":
    print(*ensure_all(), sep="\n")
