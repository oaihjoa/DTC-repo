"""Adapter gates (CPU, synthetic): the adapted models behind the shared
forward convention (dkt, dkvmn, deep_irt, akt, qikt, ukt).

  1. CAUSALITY probe (every adapter): mutating rows >= k leaves predictions
     at positions 1..k-1 bit-identical — position t uses history < t only
  2. DKT gather parity: adapter output == manual pyKT-style shifted
     one-hot gather of the pyKT net's raw output
  3. Base-training smoke (1 epoch each) + DTC post_train smoke (2 iters
     each, prop+kl) -> finite metrics with fkl present

Run: python tests/test_model_adapters.py    (from the repo root)
"""
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from tests.synth_data import ensure_all                       # noqa: E402
from eval_common import make_dcur                             # noqa: E402
from train.model_adapters import MODELS                       # noqa: E402

CSV, GROUPS, BASE = ensure_all()
OUT = ROOT / "tests" / "synth"
TINY = {
    "dkt": dict(n_question=20, n_pid=40, emb_size=16, dropout=0.1),
    "deep_irt": dict(n_question=20, n_pid=40, dim_s=16, size_m=5,
                     dropout=0.2),
    "dkvmn": dict(n_question=20, n_pid=40, dim_s=16, size_m=5,
                  dropout=0.2),
    "akt": dict(n_question=20, n_pid=40, d_model=16, n_blocks=1,
                dropout=0.05, d_ff=16),
    "qikt": dict(n_question=20, n_pid=40, emb_size=16, dropout=0.4,
                 mlp_layer_num=2),
    "ukt": dict(n_question=20, n_pid=40, d_model=16, n_blocks=1,
                dropout=0.2, d_ff=16, num_attn_heads=2, seq_len=200),
}


def run(name, cmd):
    print(f"--- {name}")
    proc = subprocess.run([sys.executable] + cmd, cwd=ROOT,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-3000:]); print(proc.stderr[-3000:])
        raise SystemExit(f"FAILED: {name}")
    return proc.stdout


def main():
    torch.manual_seed(0)
    q = torch.randint(0, 40, (2, 200))
    c = torch.randint(0, 20, (2, 200))
    r = torch.randint(0, 2, (2, 200))

    for name, cfg in TINY.items():
        model = MODELS[name](**cfg).eval()
        with torch.no_grad():
            p_full = model(make_dcur(q, c, r))
        assert p_full.shape == (2, 200), (name, p_full.shape)
        # 1. causality: rewrite everything from row k on
        k = 100
        q2, c2, r2 = q.clone(), c.clone(), r.clone()
        q2[:, k:] = torch.randint(0, 40, (2, 200 - k))
        c2[:, k:] = torch.randint(0, 20, (2, 200 - k))
        r2[:, k:] = 1 - r2[:, k:]
        with torch.no_grad():
            p_mut = model(make_dcur(q2, c2, r2))
        assert torch.equal(p_full[:, 1:k], p_mut[:, 1:k]), \
            f"{name}: future rows leaked into past predictions"
        print(f"ok: {name} causal (positions 1..{k-1} bit-identical)")

    # 2. DKT gather parity vs the pyKT net's raw output
    dkt = MODELS["dkt"](**TINY["dkt"]).eval()
    with torch.no_grad():
        y = dkt.net(c, r)
        manual = torch.gather(y[:, :-1], 2,
                              c[:, 1:].unsqueeze(-1)).squeeze(-1)
        p = dkt(make_dcur(q, c, r))
    assert torch.equal(p[:, 1:], manual), "DKT gather mismatch"
    assert torch.all(p[:, 0] == 0.5)
    print("ok: DKT adapter == pyKT shifted one-hot gather")

    # 3. smokes: base train + DTC post-train per model
    for name, cfg in TINY.items():
        cfg_json = json.dumps({k: v for k, v in cfg.items()
                               if k not in ("n_question", "n_pid")})
        base_out = OUT / f"smoke_{name}_base.pt"
        run(f"{name} base train", ["train/train_base.py", "--model",
            name, "--csv", str(CSV), "--dataset", "peiyou", "--model-cfg",
            cfg_json, "--epochs", "1", "--batch-size", "16",
            "--out", str(base_out)])
        pt_out = OUT / f"smoke_{name}_dtc.pt"
        run(f"{name} dtc post_train", ["dtc/post_train.py", "--csv",
            str(CSV), "--dataset", "peiyou", "--model", name,
            "--model-cfg", cfg_json, "--ckpt", str(base_out),
            "--loss", "prop+kl", "--iters", "2", "--eval-every", "2",
            "--batch", "8", "--n-train", "30", "--n-eval", "10",
            "--out", str(pt_out)])
        m = json.loads(pt_out.with_suffix(".final.json")
                       .read_text())["final"]
        assert "fkl" in m and all(v != float("inf") for v in m.values())
        print(f"    {name}: {json.dumps(m)}")

    print("PASS: model adapters gated + smoked")


if __name__ == "__main__":
    main()
