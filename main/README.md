# DTC: post-hoc optimization of transition consistency in knowledge tracing

An already-trained KT model (frozen `base`) is adapted post hoc to improve
transition consistency while staying close to the base predictor
DTC is the differentiable surrogate of the Transition Consistency Rate
(STCR/HTCR).

## Setup

Models and data preprocessing come from pyKT; install it at the commit the
paper used:

    pip install torch numpy pandas
    pip install git+https://github.com/pykt-team/pykt-toolkit.git@3aabdc3cab06a57e745518b070d31bcbb823af2c

Data: run pyKT's preprocessing and place its KC-expanded split file at

    data/peiyou/train_valid_sequences.csv         (XES3G5M)
    data/assist2009/train_valid_sequences.csv
    data/nips_task34/train_valid_sequences.csv

For the fairness experiment (nips_task34 only) also place the raw Eedi
`student_metadata_task_3_4.csv` and pyKT's `keyid2idx.json` in
`data/nips_task34/`; `eo/join_metadata.py` builds the groups file from them.

## Map

    train/         model_adapters.py     the 8 models behind one forward
                                         convention (pyKT classes; QIKT is a
                                         local port, train/qikt.py — its
                                         header says why)
                   train_base.py         base-model training (BCE)
                   post_train_common.py  the post-hoc loop (the method)
                   lora.py               LoRA adapters (--train-scope lora)
    dtc/           loss.py (DTC) + post_train.py
    eo/            loss.py (soft equal-opportunity gap) + post_train.py
                   + join_metadata.py
    baselines/     frappe_head.py (FRAPPE), wise_ft.py (WiSE-FT)
    evaluation/    eval_common.py (loading, scoring, AUC/NLL),
                   transition_consistency.py (HTCR/STCR),
                   evaluation_eo.py (EO gap of a checkpoint)
    experiments/   table1_pairs.sh, figure1_frontier.sh, table3_eo_stack.sh
    tests/         run_all.sh — synthetic-data checks, no dataset needed

Models: DKT, DKVMN, Deep-IRT, AKT, simpleKT, sparseKT, QIKT, UKT.

## Preservation mechanisms (paper Sec. 5.4)

    no anchor    dtc/post_train.py --loss prop
    KL (ours)    dtc/post_train.py --loss prop+kl  --beta B
    reverse KL   dtc/post_train.py --loss prop+kl  --beta B --divergence reverse_kl
    logit-MSE    dtc/post_train.py --loss prop+kl  --beta B --divergence logit_mse
    BCE          dtc/post_train.py --loss prop+bce --gamma G
    L2-SP        dtc/post_train.py --loss prop --weight-anchor l2sp --weight-alpha A
    LoRA         dtc/post_train.py --loss prop+bce --gamma 8 --train-scope lora --lora-rank R
    FRAPPE       baselines/frappe_head.py --lam L --mu 1
    WiSE-FT      baselines/wise_ft.py --base <base> --post <no-anchor ckpt>

## Reproduce

    bash experiments/table1_pairs.sh       # Table 1: 8 models x 3 datasets x 3 paired runs
    bash experiments/figure1_frontier.sh   # Figure 1 / Table 2: the sweep on QIKT
    bash experiments/table3_eo_stack.sh    # Table 3: EO -> DTC (KL vs BCE)

Each script is sequential and resumable (finished runs are skipped) and
takes env filters (`MODELS=`/`MODEL=`, `DATASETS=`, `SEEDS=`; see its
header). Checkpoints and results go to `ckpt/`: every post-hoc run writes
`<out>.final.json` with the base ("baseline") and adapted ("final")
AUC / NLL / fkl / HTCR / STCR; base training writes `<out>.train.json`.

Single runs:

    python train/train_base.py --model qikt --lr 1e-3 \
        --csv data/peiyou/train_valid_sequences.csv --dataset peiyou \
        --out ckpt/qikt_peiyou_base.pt
    python dtc/post_train.py --loss prop+kl --lam 1 --beta 64 --model qikt \
        --csv data/peiyou/train_valid_sequences.csv --dataset peiyou \
        --ckpt ckpt/qikt_peiyou_base.pt --out ckpt/qikt_peiyou_dtc_kl64.pt

## Test

    bash tests/run_all.sh

## Credits

Model implementations: pyKT (pykt-team/pykt-toolkit, MIT). HTCR/STCR:
Bai et al., AIED 2026. Baselines re-implemented from their papers: L2-SP
(Li et al., ICML 2018), LoRA (Hu et al., ICLR 2022), FRAPPE (Tifrea et
al., ICML 2024), WiSE-FT (Wortsman et al., CVPR 2022). Equal opportunity:
Hardt et al., NeurIPS 2016.
