#!/usr/bin/env bash
# Figure 1 / Table 2 (RQ2): the preservation-mechanism sweep. DTC is held
# fixed; only the preservation mechanism (or the trainable scope) varies.
# Base = the seed-1 base of Table 1 (QIKT in the paper); single seed 0.
source "$(dirname "$0")/common.sh"
MODEL=${MODEL:-qikt}
DATASETS=${DATASETS:-"peiyou assist2009 nips_task34"}
COEF="64 32 16 8 4 2"

for ds in $DATASETS; do
  train_base "$MODEL" "$ds" 1
  base=$(base_of "$MODEL" "$ds" 1)
  run() { local tag=$1; shift                   # run TAG FLAGS...
          dtc "$MODEL" "$ds" "$base" "ckpt/f1_${MODEL}_${ds}_${tag}.pt" 0 "$@"; }

  run noanchor --loss prop                                       # no anchor
  for v in $COEF; do run "kl$v"  --loss prop+kl  --beta "$v"; done      # ours
  for v in $COEF; do run "rkl$v" --loss prop+kl  --beta "$v" \
                         --divergence reverse_kl; done
  for v in $COEF; do run "bce$v" --loss prop+bce --gamma "$v"; done
  for v in $COEF 1 0.5 0.25 0.125 0.0625; do
    run "logitmse$v" --loss prop+kl --beta "$v" --divergence logit_mse; done
  for v in $COEF 1 0.5 0.25 0.125 0.0625 0.03125; do
    run "l2sp$v" --loss prop --weight-anchor l2sp --weight-alpha "$v"; done
  for r in 1 2 4 8; do
    run "lora_r$r" --loss prop+bce --gamma 8 --train-scope lora \
                   --lora-rank "$r"; done

  for lam in 0.03125 0.125 0.5 2 8 32 128 512; do                # FRAPPE
    out="ckpt/f1_${MODEL}_${ds}_frappe$lam.pt"
    once "${out%.pt}.final.json" $PY baselines/frappe_head.py \
      --csv "$(csv_of "$ds")" --dataset "$ds" --model "$MODEL" \
      --ckpt "$base" --out "$out" --lam "$lam" --mu 1 --hidden 64 \
      --lr 1e-3 --iters 1500 --batch 256 --seed 0 --eval-every 250
  done

  out="ckpt/f1_${MODEL}_${ds}_wise.json"                         # WiSE-FT
  once "$out" $PY baselines/wise_ft.py --model "$MODEL" --dataset "$ds" \
    --base "$base" --post "ckpt/f1_${MODEL}_${ds}_noanchor.pt" --out "$out"
done
