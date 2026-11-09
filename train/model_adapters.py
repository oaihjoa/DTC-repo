"""The eight models of the paper behind one forward convention.

CONVENTION (what the trainer, the post-hoc loop and the metrics assume):
forward(dcur) returns (B, T_MAX) P(correct), where position t is the
prediction for row t made from history STRICTLY BEFORE t; position 0 is
never scored. dcur is pyKT's dict of unshifted/shifted window views.

Model code comes from pyKT (install it; see README):
  simplekt, sparsekt   pyKT's classes already follow the convention and are
                       used as they are.
  dkt                  forward(c, r) returns (B, T, num_c) where the output
                       at t predicts t+1; the adapter reproduces pyKT's
                       shifted one-hot gather: p[:, t] = y[:, t-1, c_t] for
                       t >= 1; p[:, 0] = 0.5 (unscored filler).
  dkvmn, deep_irt      forward(c, r) is already row-aligned (read-before-
                       write memory): passthrough.
  akt                  forward(c, r, q) returns (preds, rasch_l2); the
                       adapter stashes the Rasch regularizer on
                       .last_reg_loss so base training adds it (pyKT's own
                       recipe); post-hoc adaptation reads only the preds.
  ukt                  pyKT-convention forward; the adapter adds the masks
                       view and UKT's contrastive training term.
  qikt                 the one model NOT taken from pyKT as is: train/qikt.py
                       is a port whose question head uses a gather instead
                       of pyKT's (B, T, num_q) one-hot, which runs out of
                       memory at num_q = 17737 (ASSISTments2009).
"""
import torch
import torch.nn as nn


def _capture_output(module, run_forward):
    """Run run_forward() with a forward hook on `module`, returning
    (forward result, module's captured output) — feature access without
    touching the model's source."""
    captured = []
    handle = module.register_forward_hook(
        lambda mod, inputs, output: captured.append(output))
    try:
        result = run_forward()
    finally:
        handle.remove()
    return result, captured[0]


def _rebuild_window(dcur):
    """Restore the full unshifted (B, T) window from pyKT's split views —
    same reconstruction as the simpleKT rewrite."""
    q = torch.cat((dcur["qseqs"].long()[:, 0:1],
                   dcur["shft_qseqs"].long()), dim=1)
    c = torch.cat((dcur["cseqs"].long()[:, 0:1],
                   dcur["shft_cseqs"].long()), dim=1)
    r = torch.cat((dcur["rseqs"].long()[:, 0:1],
                   dcur["shft_rseqs"].long()), dim=1)
    return q, c, r


def _lstm_predict_with_features(adapter, dcur):
    """(p, features) for the FRAPPE head on the DKT-family adapters.
    Features at row t must be CAUSAL for the correction to remain a valid
    KT model: [h_{t-1} | response-free concept query], where h is the LSTM
    output the readout consumes (dropout is identity in the eval-mode
    convention) — h_t encodes interactions <= t, so it is shifted right one
    step (h_{-1} = 0; row 0 is never scored). The query is the mean of the
    (c_t, r=0) and (c_t, r=1) interaction embeddings: concept identity
    without the current response."""
    _, c, r = _rebuild_window(dcur)
    p, h = _capture_output(adapter.net.lstm_layer,
                           lambda: adapter.forward(dcur))
    h = h[0]                                  # LSTM returns (output, state)
    h_prev = torch.cat((torch.zeros_like(h[:, :1]), h[:, :-1]), dim=1)
    query = 0.5 * (adapter.net.interaction_emb(c)
                   + adapter.net.interaction_emb(c + adapter.net.num_c))
    return p, torch.cat((h_prev, query), dim=-1)


def _memory_predict_with_features(adapter, dcur):
    """(p, f) for the DKVMN-family adapters. f_t = tanh(f_layer([read_t |
    k_t])) is read-before-write (memory of history < t) fused with the
    current question key — causal and row-aligned with p, and exactly the
    tensor the model's own qtest branch would return (captured by hook so
    deep_irt's qtest print never fires)."""
    p, f_linear = _capture_output(adapter.net.f_layer,
                                  lambda: adapter.forward(dcur))
    return p, torch.tanh(f_linear)


class DKTAdapter(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        from pykt.models.dkt import DKT
        self.net = DKT(num_c=cfg["n_question"], emb_size=cfg["emb_size"],
                       dropout=cfg["dropout"])
        self.frappe_feature_dim = 2 * cfg["emb_size"]

    def predict_with_features(self, dcur):
        return _lstm_predict_with_features(self, dcur)

    def forward(self, dcur):
        _, c, r = _rebuild_window(dcur)
        y = self.net(c, r)                       # (B, T, num_c)
        # pyKT's convention: output at t predicts t+1 -> gather the NEXT
        # concept from the PREVIOUS position's output (their cshft gather)
        p_next = torch.gather(y[:, :-1], 2,
                              c[:, 1:].unsqueeze(-1)).squeeze(-1)
        filler = torch.full_like(p_next[:, :1], 0.5)
        return torch.cat((filler, p_next), dim=1)


class DKVMNAdapter(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        from pykt.models.dkvmn import DKVMN
        self.net = DKVMN(num_c=cfg["n_question"], dim_s=cfg["dim_s"],
                         size_m=cfg["size_m"], dropout=cfg["dropout"])
        self.frappe_feature_dim = cfg["dim_s"]

    def predict_with_features(self, dcur):
        return _memory_predict_with_features(self, dcur)

    def forward(self, dcur):
        _, c, r = _rebuild_window(dcur)
        return self.net(c, r)                    # already (B, T), row-aligned


class AKTAdapter(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        from pykt.models.akt import AKT
        self.net = AKT(n_question=cfg["n_question"], n_pid=cfg["n_pid"],
                       d_model=cfg["d_model"], n_blocks=cfg["n_blocks"],
                       dropout=cfg["dropout"], d_ff=cfg["d_ff"])
        self.last_reg_loss = 0.0

    def forward(self, dcur):
        q, c, r = _rebuild_window(dcur)
        preds, reg = self.net(c, r, q)
        self.last_reg_loss = reg
        return preds


class DeepIRTAdapter(nn.Module):
    """Deep-IRT (DKVMN + IRT ability/difficulty heads): row-aligned
    passthrough like DKVMN — the psychometric-structure model."""

    def __init__(self, **cfg):
        super().__init__()
        from pykt.models.deep_irt import DeepIRT
        self.net = DeepIRT(num_c=cfg["n_question"], dim_s=cfg["dim_s"],
                           size_m=cfg["size_m"], dropout=cfg["dropout"])
        self.frappe_feature_dim = cfg["dim_s"]

    def predict_with_features(self, dcur):
        return _memory_predict_with_features(self, dcur)

    def forward(self, dcur):
        _, c, r = _rebuild_window(dcur)
        return self.net(c, r)


class QIKTAdapter(nn.Module):
    """QIKT (Chen et al., AAAI 2023; train/qikt.py). The net consumes the full (q, c, r) window and
    predicts rows 1..T-1 (LSTMs over interactions [:-1] against the
    shifted question identity); the adapter supplies the qshft/cshft
    views the heads gather with (single-KC rows -> trailing concept
    axis on cshft) and returns the official 'an'-fused y with the
    DKT-style 0.5 filler at row 0. Official default loss lambdas are 0,
    so the official training objective IS plain BCE on this output —
    no last_reg_loss hook needed."""

    def __init__(self, **cfg):
        super().__init__()
        from train.qikt import QIKTNet
        self.net = QIKTNet(num_q=cfg["n_pid"], num_c=cfg["n_question"],
                           emb_size=cfg["emb_size"],
                           dropout=cfg["dropout"],
                           mlp_layer_num=cfg["mlp_layer_num"])
        self.frappe_feature_dim = 4 * cfg["emb_size"]

    def predict_with_features(self, dcur):
        """(p, f) for the FRAPPE head. f_t = [que_h | concept_h | emb_qc]:
        both LSTMs run over interactions [:-1], so their row j encodes
        history <= j and is consumed to predict row j+1 — already the
        h_{t-1} shift; emb_qc[:, 1:] is the response-free question+concept
        identity of the predicted row (pre-fusion, 2*emb wide). Row 0
        (the 0.5 filler, never scored) gets a zero feature row. LSTM
        outputs are captured pre-dropout; dropout is identity under the
        eval-mode convention."""
        caps = {}
        hooks = [self.net.que_lstm_layer.register_forward_hook(
                     lambda m, i, o: caps.__setitem__("qh", o[0])),
                 self.net.concept_lstm_layer.register_forward_hook(
                     lambda m, i, o: caps.__setitem__("ch", o[0])),
                 self.net.que_emb.register_forward_hook(
                     lambda m, i, o: caps.__setitem__("qc", o[2]))]
        try:
            p = self.forward(dcur)
        finally:
            for h in hooks:
                h.remove()
        f = torch.cat((caps["qh"], caps["ch"], caps["qc"][:, 1:, :]),
                      dim=-1)
        f = torch.cat((torch.zeros_like(f[:, :1]), f), dim=1)
        return p, f

    def forward(self, dcur):
        q, c, r = _rebuild_window(dcur)
        data = {"qshft": q[:, 1:], "cshft": c[:, 1:].unsqueeze(-1)}
        outputs = self.net(q, c, r.float(), data=data)
        p_next = self.net.fused_y(outputs)
        filler = torch.full_like(p_next[:, :1], 0.5)
        return torch.cat((filler, p_next), dim=1)


class UKTAdapter(nn.Module):
    """UKT (Cheng et al., AAAI 2025; pyKT's class, emb_type stoc_qid =
    the UKT-paper model). UKT is pyKT-convention: its
    forward consumes the dcur dict directly and returns row-aligned (B, T)
    predictions like simpleKT (mean-of-Gaussian readout through the output
    head). The adapter passes dcur through, adding the 'masks' view the
    upstream signature reads (used only on the training path)."""

    def __init__(self, **cfg):
        super().__init__()
        from pykt.models.ukt import UKT
        # use_CL: UKT's own training objective, BCE + cl_weight *
        # WassersteinCL (cl_weight 0.02, uncertainty aug on). It is only
        # active in training mode and adds no parameters.
        self.net = UKT(n_question=cfg["n_question"], n_pid=cfg["n_pid"],
                       d_model=cfg["d_model"], n_blocks=cfg["n_blocks"],
                       dropout=cfg["dropout"], d_ff=cfg["d_ff"],
                       num_attn_heads=cfg["num_attn_heads"],
                       seq_len=cfg.get("seq_len", 200),
                       emb_type="stoc_qid", use_CL=True)
        self.frappe_feature_dim = 4 * cfg["d_model"]
        self.last_reg_loss = 0.0

    @staticmethod
    def _uncertainty_aug(dcur):
        """Official UKTDataset need_aug construction, verbatim SEMANTICS:
        their per-sample loop re-reads index 0 inside the loop, so after
        i=0 flips it the condition self-disables — the net effect is
        exactly "flip position 0 iff it equals the last response", for
        both rshft_aug and r_aug. Vectorized here (equivalence is the
        short-circuit argument above). Deviation (documented): the
        official code indexes the last response under the real pad mask;
        our adapter path sees pre-clamped windows, so the all-True
        window mask is used — same convention our UKT bases already
        train under."""
        r = dcur["rseqs"].clone()
        rs = dcur["shft_rseqs"].clone()
        last = rs[:, -1:]
        rs[:, :1] = torch.where(rs[:, :1] == last, 1 - rs[:, :1], rs[:, :1])
        r[:, :1] = torch.where(r[:, :1] == last, 1 - r[:, :1], r[:, :1])
        return r, rs

    def forward(self, dcur):
        if "masks" not in dcur:
            dcur = dict(dcur)
            dcur["masks"] = torch.ones_like(
                dcur["shft_rseqs"], dtype=torch.bool)
        if self.training:
            d = dict(dcur)
            d["r_aug"], d["shft_r_aug"] = self._uncertainty_aug(d)
            preds, sim, _y2, _y3, _temp = self.net(d, train=True)
            # rides the trainer's published-aux hook (AKT Rasch precedent);
            # official total: BCE + cl_weight * sim
            self.last_reg_loss = self.net.cl_weight * sim
            return preds
        self.last_reg_loss = 0.0
        return self.net(dcur)

    def predict_with_features(self, dcur):
        """(p, features) for the FRAPPE head. features = UKT's own qtest
        concat_q = [mean_d_output | cov_d_output | q_mean_embed |
        q_cov_embed] (B, T, 4*d_model). CAUSALITY AUDITED: flipping r_t changes preds/concat_q ONLY at positions
        > t (machine-precision zero at <= t), so row t is history-only
        plus a response-free query — the same contract as the
        simpleKT/DKT-family feature hooks."""
        if "masks" not in dcur:
            dcur = dict(dcur)
            dcur["masks"] = torch.ones_like(
                dcur["shft_rseqs"], dtype=torch.bool)
        return self.net(dcur, qtest=True)


def _simplekt(**cfg):
    from pykt.models.simplekt import simpleKT
    return simpleKT(**cfg)


def _sparsekt(**cfg):
    from pykt.models.sparsekt import sparseKT
    return sparseKT(**cfg)


# model name -> constructor(**cfg); cfg = MODEL_DEFAULTS + DATASET_CFG
# (evaluation/eval_common.py)
MODELS = {"simplekt": _simplekt, "sparsekt": _sparsekt,
          "dkt": DKTAdapter, "dkvmn": DKVMNAdapter, "akt": AKTAdapter,
          "deep_irt": DeepIRTAdapter, "qikt": QIKTAdapter,
          "ukt": UKTAdapter}
