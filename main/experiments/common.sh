# Shared settings for the paper's experiment scripts. Source, don't run.
# Everything runs sequentially — one model at a time.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=${PY:-python3}
# post-hoc recipe shared by every trained method (paper Sec. 5.4)
POST="--lam 1.0 --lr 5e-6 --iters 1500 --batch 256 --eval-every 250"
# base-model recipe (paper Sec. 5.3)
BASE_RECIPE="--epochs 500 --min-delta 0 --patience 20 --lr-schedule plateau \
--valid-protocol aligned --ema-horizon-epochs 3"

csv_of() { echo "data/$1/train_valid_sequences.csv"; }
base_of() { echo "ckpt/base_s$3_$1_$2.pt"; }          # model dataset seed

lr_of() {                                              # per-model base lr
  case "$1" in
    dkt|dkvmn|deep_irt|qikt) echo 1e-3 ;;
    simplekt|sparsekt|ukt)   echo 1e-4 ;;
    akt)                     echo 1e-5 ;;
    *) echo "unknown model $1" >&2; exit 1 ;;
  esac
}

# once MARKER CMD... : run CMD unless MARKER already exists (resumable)
once() {
  local marker=$1; shift
  if [ -s "$marker" ]; then echo "skip (done): $marker"; return; fi
  echo "+ $*"; "$@"
}

train_base() {                                         # model dataset seed
  local model=$1 ds=$2 seed=$3 out; out=$(base_of "$1" "$2" "$3")
  local extra=()
  # UKT trains with its own contrastive objective (doubles activation memory)
  [ "$model" = ukt ] && extra=(--batch-size 128 --model-cfg '{"use_cl": 1}')
  once "${out%.pt}.train.json" $PY train/train_base.py --model "$model" \
    --csv "$(csv_of "$ds")" --dataset "$ds" --out "$out" \
    --lr "$(lr_of "$model")" --seed "$seed" $BASE_RECIPE \
    ${extra[@]+"${extra[@]}"}
}

# dtc MODEL DATASET CKPT OUT SEED FLAGS... : one post-hoc DTC run
dtc() {
  local model=$1 ds=$2 ckpt=$3 out=$4 seed=$5; shift 5
  once "${out%.pt}.final.json" $PY dtc/post_train.py \
    --csv "$(csv_of "$ds")" --dataset "$ds" --model "$model" \
    --ckpt "$ckpt" --out "$out" --seed "$seed" $POST "$@"
}
