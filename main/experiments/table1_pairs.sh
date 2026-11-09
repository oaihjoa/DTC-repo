#!/usr/bin/env bash
# Table 1 (RQ1): post-hoc DTC with the KL anchor, 8 models x 3 datasets,
# three PAIRED runs — seed s trains a fresh base and then adapts THAT base.
#   base:  next-response BCE, the Sec. 5.3 recipe
#   post:  L = (1 - DTC) + 64 * KL(base || theta)
# Filter with env vars, e.g.  MODELS="qikt" DATASETS="nips_task34" bash $0
source "$(dirname "$0")/common.sh"
MODELS=${MODELS:-"simplekt ukt dkt dkvmn deep_irt akt sparsekt qikt"}
DATASETS=${DATASETS:-"peiyou assist2009 nips_task34"}
SEEDS=${SEEDS:-"1 2 3"}

for model in $MODELS; do for ds in $DATASETS; do for seed in $SEEDS; do
  train_base "$model" "$ds" "$seed"
  dtc "$model" "$ds" "$(base_of "$model" "$ds" "$seed")" \
      "ckpt/t1_${model}_${ds}_s${seed}_kl64.pt" "$seed" \
      --loss prop+kl --beta 64
done; done; done
