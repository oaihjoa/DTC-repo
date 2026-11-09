"""End-to-end smoke on synthetic data (CPU, about a minute): every entry
point runs a real forward/backward and writes its artifacts.

  1. train/train_base.py --model simplekt / sparsekt (1 epoch, tiny cfg)
  2. the paper's base recipe flags (plateau / aligned / EMA) -> trains
  3. dtc/ and eo/ post_train.py --loss prop+kl, 2 iters -> finite metrics
  4. the Table-3 chain: base -> EO -> DTC (KL and BCE branches) -> runs
  5. evaluation/evaluation_eo.py on the tiny base -> valid JSON

This is a SMOKE (finite losses, artifacts exist, JSON parses) — numeric
checks live in test_losses.py.

Run: python tests/test_train_smoke.py    (from the repo root)
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.synth_data import ensure_all, TINY_SIMPLEKT, TINY_SPARSEKT  # noqa: E402

CSV, GROUPS, BASE = ensure_all()
CFG_S = json.dumps(TINY_SIMPLEKT)
CFG_K = json.dumps(TINY_SPARSEKT)


def run(name, cmd):
    print(f"--- {name}")
    proc = subprocess.run([sys.executable] + cmd, cwd=ROOT,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise SystemExit(f"FAILED: {name}")
    return proc.stdout


def finite_final(out):
    final = json.loads(out.with_suffix(".final.json").read_text())["final"]
    for k, v in final.items():
        assert v == v and abs(v) != float("inf"), (out.name, k, v)
    return final


def main():
    out_dir = ROOT / "tests" / "synth"

    for model, cfg in (("simplekt", CFG_S), ("sparsekt", CFG_K)):
        run(f"train_base {model}",
            ["train/train_base.py", "--model", model, "--csv", str(CSV),
             "--dataset", "peiyou", "--model-cfg", cfg, "--epochs", "1",
             "--batch-size", "16",
             "--out", str(out_dir / f"smoke_{model}.pt")])
        assert (out_dir / f"smoke_{model}.pt").exists()

    run("train_base simplekt (paper base recipe flags)",
        ["train/train_base.py", "--model", "simplekt", "--csv", str(CSV),
         "--dataset", "peiyou", "--model-cfg", CFG_S, "--epochs", "2",
         "--batch-size", "16", "--min-delta", "0", "--patience", "20",
         "--lr-schedule", "plateau", "--valid-protocol", "aligned",
         "--ema-horizon-epochs", "3",
         "--out", str(out_dir / "smoke_simplekt_recipe.pt")])
    assert (out_dir / "smoke_simplekt_recipe.train.json").exists()

    # flags shared by every post-training arm; each arm adds its own head
    common = ["--csv", str(CSV), "--dataset", "peiyou",
              "--model", "simplekt", "--model-cfg", CFG_S,
              "--iters", "2", "--eval-every", "2",
              "--batch", "8", "--n-train", "30", "--n-eval", "10"]
    arms = {
        "dtc": ["dtc/post_train.py"],
        "eo": ["eo/post_train.py", "--groups", str(GROUPS)],
    }
    for name, head in arms.items():
        out = out_dir / f"smoke_{name}.pt"
        run(f"post_train {name}", head + common + [
            "--loss", "prop+kl", "--ckpt", str(BASE), "--out", str(out)])
        print(f"    final: {json.dumps(finite_final(out))}")
    assert {"htcr", "stcr", "fkl", "auc"} <= set(
        finite_final(out_dir / "smoke_dtc.pt"))
    assert {"eo", "fkl", "auc"} <= set(
        finite_final(out_dir / "smoke_eo.pt"))

    # Table-3 chain: DTC from the EO-adapted checkpoint, anchored to it
    for branch, flags in {"kl": ["--loss", "prop+kl", "--beta", "64"],
                          "bce": ["--loss", "prop+bce", "--gamma", "64"]
                          }.items():
        out = out_dir / f"smoke_eo_then_dtc_{branch}.pt"
        run(f"EO -> DTC ({branch})", ["dtc/post_train.py"] + common + flags
            + ["--ckpt", str(out_dir / "smoke_eo.pt"), "--out", str(out)])
        finite_final(out)

    stdout = run("evaluation_eo",
                 ["evaluation/evaluation_eo.py", "--csv", str(CSV),
                  "--dataset", "peiyou", "--model", "simplekt",
                  "--model-cfg", CFG_S, "--ckpt", str(BASE),
                  "--groups", str(GROUPS)])
    # the script prints its metrics JSON as the last stdout line
    parsed = json.loads(stdout.strip().splitlines()[-1])
    assert "eo" in parsed, parsed
    print(f"    eo audit: {json.dumps(parsed)}")

    print("PASS: all entry points ran end-to-end on synthetic data")


if __name__ == "__main__":
    main()
