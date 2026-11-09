# DTC: post-hoc optimization of transition consistency in knowledge tracing

Adapt an already-trained KT model to improve transition consistency while
staying close to its original predictions. Accepted to [WSDM 2027 Findings](https://www.wsdm-conference.org/2027/)

## Setup

    pip install torch numpy pandas
    pip install git+https://github.com/pykt-team/pykt-toolkit.git@3aabdc3cab06a57e745518b070d31bcbb823af2c

Put pyKT's preprocessed split at `data/<dataset>/train_valid_sequences.csv`.

## Run

Train a base model, then adapt it with DTC:

    python train/train_base.py --model qikt --dataset peiyou --out ckpt/base.pt
    python dtc/post_train.py --model qikt --dataset peiyou --ckpt ckpt/base.pt --beta 64 --out ckpt/dtc.pt

`--model`: `dkt dkvmn deep_irt akt simplekt sparsekt qikt ukt`
`--dataset`: `peiyou` (XES3G5M), `assist2009`, `nips_task34`
`--beta`: strength of the KL anchor to the base model

The result is in `ckpt/dtc.final.json`: AUC, NLL, HTCR and STCR before
("baseline") and after ("final").

## Credits

- Models and data preprocessing: [pyKT](https://github.com/pykt-team/pykt-toolkit) ([paper](https://arxiv.org/abs/2206.11460))
- Transition consistency (HTCR/STCR): [Bai et al., AIED 2026](https://doi.org/10.1007/978-3-032-29744-0_34)
- Baselines: [L2-SP](https://arxiv.org/abs/1802.01483), [logit-MSE](https://arxiv.org/abs/1312.6184), [LoRA](https://arxiv.org/abs/2106.09685), [FRAPPE](https://arxiv.org/abs/2312.02592), [WiSE-FT](https://arxiv.org/abs/2109.01903)
- Equal opportunity: [Hardt et al., 2016](https://arxiv.org/abs/1610.02413)
