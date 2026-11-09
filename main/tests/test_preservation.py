"""Gates + smokes for the preservation mechanisms the paper compares
(Sec. 5.4), on synthetic data (CPU).

  1. anchor_divergence unit checks vs hand formulas (float64)
  2. DEFAULT-PATH GUARD: post_train with no optional flags == post_train
     with the explicit defaults, identical final metrics
  3. LoRA gate (qikt): wrapped model bit-exact to base at init; only
     adapters trainable
  4. 2-iter smokes, finite metrics: no anchor, BCE, reverse KL, logit-MSE,
     L2-SP (simplekt) and LoRA (qikt)
  5. FRAPPE head (qikt): 2-iter smoke, finite metrics, fkl present
  6. WiSE-FT: the alpha = 1 row reproduces the fine-tuned endpoint

Run: python tests/test_preservation.py    (from the repo root)
"""
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from tests.synth_data import (ensure_all, ensure_qikt_base,
                              TINY_SIMPLEKT, TINY_QIKT)      # noqa: E402

CSV, GROUPS, BASE = ensure_all()
QIKT_BASE = ensure_qikt_base()
CFG = json.dumps(TINY_SIMPLEKT)
CFG_Q = json.dumps(TINY_QIKT)
SMOKE = ["--iters", "2", "--eval-every", "2", "--batch", "8",
         "--n-train", "30", "--n-eval", "10"]
OUT = ROOT / "tests" / "synth"


def run(name, cmd):
    print(f"--- {name}")
    proc = subprocess.run([sys.executable] + cmd, cwd=ROOT,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-3000:]); print(proc.stderr[-3000:])
        raise SystemExit(f"FAILED: {name}")
    return proc.stdout


def final_metrics(out_path):
    return json.loads(Path(out_path).with_suffix(".final.json")
                      .read_text())["final"]


def all_finite(metrics):
    return all(v == v and abs(v) != float("inf") for v in metrics.values())


def main():
    # ---- 1. divergence unit checks -----------------------------------
    from train.post_train_common import anchor_divergence
    p = torch.tensor([0.2, 0.7], dtype=torch.float64)
    q = torch.tensor([0.4, 0.6], dtype=torch.float64)
    fkl = p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()
    rkl = q * (q / p).log() + (1 - q) * ((1 - q) / (1 - p)).log()
    lmse = (torch.logit(q) - torch.logit(p)) ** 2
    assert torch.allclose(anchor_divergence("forward_kl", p, q), fkl)
    assert torch.allclose(anchor_divergence("reverse_kl", p, q), rkl)
    assert torch.allclose(anchor_divergence("logit_mse", p, q), lmse)
    print("ok: divergence formulas")

    # ---- 2. default-path guard ---------------------------------------
    common = ["dtc/post_train.py", "--csv", str(CSV), "--dataset",
              "peiyou", "--model", "simplekt", "--model-cfg", CFG,
              "--ckpt", str(BASE)] + SMOKE
    common_qikt = ["dtc/post_train.py", "--csv", str(CSV), "--dataset",
                   "peiyou", "--model", "qikt", "--model-cfg", CFG_Q,
                   "--ckpt", str(QIKT_BASE)] + SMOKE
    run("default flags", common + ["--loss", "prop+kl",
                                   "--out", str(OUT / "p_default.pt")])
    run("explicit defaults", common +
        ["--loss", "prop+kl", "--divergence", "forward_kl",
         "--train-scope", "all", "--weight-anchor", "none",
         "--out", str(OUT / "p_explicit.pt")])
    m_def = final_metrics(OUT / "p_default.pt")
    m_exp = final_metrics(OUT / "p_explicit.pt")
    assert m_def == m_exp, (m_def, m_exp)
    print("ok: explicit defaults reproduce the default path exactly")

    # ---- 3. LoRA gate (in-process) -----------------------------------
    from eval_common import build_model, make_dcur, forward_preds
    from train.lora import inject_lora
    torch.manual_seed(0)
    model_a = build_model("qikt", "peiyou", str(QIKT_BASE), CFG_Q)
    model_b = build_model("qikt", "peiyou", str(QIKT_BASE), CFG_Q)
    trainable = inject_lora(model_b, rank=8)
    q = torch.randint(0, 40, (2, 200)); c = torch.randint(0, 20, (2, 200))
    r = torch.randint(0, 2, (2, 200))
    with torch.no_grad():
        out_a = forward_preds(model_a, make_dcur(q, c, r))
        out_b = forward_preds(model_b, make_dcur(q, c, r))
    assert torch.equal(out_a, out_b), "LoRA not bit-exact at init"
    names = [n for n, p in model_b.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names), names
    assert len(trainable) == len(names)
    print(f"ok: LoRA bit-exact at init; {len(names)} adapter tensors "
          f"trainable, base frozen")

    # ---- 4. one smoke per trained preservation mechanism -------------
    for name, cmd in {
        "noanchor": common + ["--loss", "prop"],
        "bce": common + ["--loss", "prop+bce", "--gamma", "8"],
        "reverse_kl": common + ["--loss", "prop+kl",
                                "--divergence", "reverse_kl"],
        "logit_mse": common + ["--loss", "prop+kl",
                               "--divergence", "logit_mse"],
        "l2sp": common + ["--loss", "prop", "--weight-anchor", "l2sp",
                          "--weight-alpha", "15"],
        "lora": common_qikt + ["--loss", "prop+bce", "--gamma", "8",
                               "--train-scope", "lora", "--lora-rank", "4",
                               "--lr", "1e-4"],
    }.items():
        out = OUT / f"p_{name}.pt"
        run(f"arm {name}", cmd + ["--out", str(out)])
        m = final_metrics(out)
        assert all_finite(m), (name, m)
        print(f"    {name}: {json.dumps(m)}")

    # ---- 5. FRAPPE head smoke ----------------------------------------
    run("frappe head", ["baselines/frappe_head.py", "--csv", str(CSV),
        "--dataset", "peiyou", "--model", "qikt", "--model-cfg", CFG_Q,
        "--ckpt", str(QIKT_BASE), "--lam", "2", "--mu", "1",
        "--out", str(OUT / "p_frappe.pt")] + SMOKE)
    m = final_metrics(OUT / "p_frappe.pt")
    assert "fkl" in m and all_finite(m), m
    print("ok: frappe head")

    # ---- 6. WiSE-FT: alpha = 1 is the fine-tuned endpoint ------------
    wise_out = OUT / "p_wise.json"
    run("wise-ft", ["baselines/wise_ft.py", "--model", "simplekt",
        "--dataset", "peiyou", "--csv", str(CSV), "--model-cfg", CFG,
        "--base", str(BASE), "--post", str(OUT / "p_noanchor.pt"),
        "--n-eval", "10", "--out", str(wise_out)])
    rows = json.loads(wise_out.read_text())["rows"]
    assert len(rows) == 9, sorted(rows)          # endpoint + 8 alphas
    for k, v in rows["post_endpoint"].items():
        assert abs(rows["wise1.0"][k] - v) < 1e-6, (k, rows["wise1.0"][k], v)
    assert all(all_finite(row) for row in rows.values())
    print("ok: wise-ft alpha=1 reproduces the endpoint")

    print("PASS: all preservation mechanisms gated + smoked")


if __name__ == "__main__":
    main()
