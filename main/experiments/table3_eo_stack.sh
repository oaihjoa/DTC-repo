#!/usr/bin/env bash
# Table 3 (RQ3): does post-hoc DTC preserve a previously installed property?
# NIPS34 only (the dataset with a demographic attribute), three paired runs.
#   stage 1  base -> +EO          L = 25 * EOgap + 8 * KL(base || theta)
#   stage 2  +EO  -> +DTC, KL     L = (1 - DTC) + 64 * KL(EO model || theta)
#            +EO  -> +DTC, BCE    L = (1 - DTC) + 64 * BCE
#   audit    evaluation/evaluation_eo.py on all four checkpoints
# Needs data/groups_nips_task34.json (built here on first run).
source "$(dirname "$0")/common.sh"
MODELS=${MODELS:-"qikt deep_irt simplekt"}
SEEDS=${SEEDS:-"1 2 3"}
DS=nips_task34; GROUPS_JSON=data/groups_nips_task34.json
[ -s "$GROUPS_JSON" ] || $PY eo/join_metadata.py

for model in $MODELS; do for seed in $SEEDS; do
  train_base "$model" "$DS" "$seed"
  base=$(base_of "$model" "$DS" "$seed")
  stem="ckpt/t3_${model}_s${seed}"

  once "${stem}_eo.final.json" $PY eo/post_train.py \
    --csv "$(csv_of $DS)" --dataset $DS --model "$model" --ckpt "$base" \
    --out "${stem}_eo.pt" --loss prop+kl --lam 25 --beta 8 \
    --groups $GROUPS_JSON --group-attr gender \
    --lr 5e-6 --iters 1500 --batch 256 --seed "$seed" --eval-every 250

  dtc "$model" $DS "${stem}_eo.pt" "${stem}_eo_kl64.pt"  "$seed" \
      --loss prop+kl  --beta 64
  dtc "$model" $DS "${stem}_eo.pt" "${stem}_eo_bce64.pt" "$seed" \
      --loss prop+bce --gamma 64

  for stage in base eo eo_kl64 eo_bce64; do
    ckpt="${stem}_${stage}.pt"; [ "$stage" = base ] && ckpt=$base
    out="${stem}_eoaudit_${stage}.json"
    [ -s "$out" ] && { echo "skip (done): $out"; continue; }
    $PY evaluation/evaluation_eo.py --csv "$(csv_of $DS)" --dataset $DS \
      --model "$model" --ckpt "$ckpt" --groups $GROUPS_JSON --group-attr gender \
      > "$out.tmp" && mv "$out.tmp" "$out"
  done
done; done
