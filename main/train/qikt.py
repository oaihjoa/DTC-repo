"""QIKT network (Chen et al., AAAI 2023 — question-centric interpretable
KT), ported from pyKT (pykt-team/pykt-toolkit, MIT): pykt/models/qikt.py
(MLP, QIKTNet, get_outputs, get_avg_fusion_concepts) and
pykt/models/que_base_model.py (QueEmb 'iekt' branch, get_avg_skill_emb).

WHY A PORT instead of importing pyKT's class — one change that matters:
get_outputs selects the question head with a gather instead of pyKT's
(y_all * one_hot(qshft)).sum(-1). The result is bit-identical, but the
one-hot materializes a (B, T-1, num_q) tensor that runs out of memory at
num_q = 17737 (ASSISTments2009). Everything else is mechanical (imports,
device handling, single-KC rows); the network math is line-identical.

Official wandb_qikt_train.py defaults: emb_type 'iekt', emb_size 300,
dropout 0.4, mlp_layer_num 2, loss_*_lambda ALL 0 (=> the official
training objective is plain BCE on the fused output y),
output_{q_all,c_all,c_next}_lambda = 1, output_q_next_lambda = 0,
output_mode 'an'.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
                                   for _ in range(n_layer)])
        self.dropout = nn.Dropout(p=dpo)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))


class QueEmbIEKT(nn.Module):
    """QueEmb restricted to the 'iekt' branch QIKT uses."""

    def __init__(self, num_q, num_c, emb_size, device="cpu"):
        super().__init__()
        self.device = device
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.que_emb = nn.Embedding(num_q, emb_size)
        self.concept_emb = nn.Parameter(
            torch.randn(num_c, emb_size).to(device), requires_grad=True)
        self.que_c_linear = nn.Linear(2 * emb_size, emb_size)

    def get_avg_skill_emb(self, c):
        concept_emb_cat = torch.cat(
            [torch.zeros(1, self.emb_size,
                         device=self.concept_emb.device),
             self.concept_emb], dim=0)
        related_concepts = (c + 1).long()
        concept_emb_sum = concept_emb_cat[related_concepts, :].sum(axis=-2)
        concept_num = torch.where(related_concepts != 0, 1, 0).sum(
            axis=-1).unsqueeze(-1)
        concept_num = torch.where(concept_num == 0, 1, concept_num)
        return concept_emb_sum / concept_num

    def forward(self, q, c, r):
        if c.dim() == 2:          # single-KC rows -> (B, T, 1) concept list
            c = c.unsqueeze(-1)   # (mechanical adaptation; pyKT que-level
                                  # loaders pass (B, T, max_concepts))
        emb_c = self.get_avg_skill_emb(c)
        emb_q = self.que_emb(q)
        emb_qc = torch.cat([emb_q, emb_c], dim=-1)
        xemb = self.que_c_linear(emb_qc)
        emb_qca = torch.cat(
            [emb_qc.mul((1 - r).unsqueeze(-1).repeat(1, 1,
                                                     self.emb_size * 2)),
             emb_qc.mul(r.unsqueeze(-1).repeat(1, 1, self.emb_size * 2))],
            dim=-1)
        return xemb, emb_qca, emb_qc, emb_q, emb_c


def get_outputs(net, emb_qc_shift, h, data, add_name="",
                model_type="question"):
    outputs = {}
    if model_type == "question":
        h_next = torch.cat([emb_qc_shift, h], axis=-1)
        y_question_next = torch.sigmoid(net.out_question_next(h_next))
        y_question_all = torch.sigmoid(net.out_question_all(h))
        outputs["y_question_next" + add_name] = y_question_next.squeeze(-1)
        # gather-based selection: bit-identical to the upstream
        # (y_all * one_hot(qshft)).sum(-1) but without materializing the
        # (B, T-1, num_q) one-hot — the upstream form OOMs at num_q=17737
        outputs["y_question_all" + add_name] = torch.gather(
            y_question_all, -1,
            data["qshft"].long().unsqueeze(-1)).squeeze(-1)
    else:
        h_next = torch.cat([emb_qc_shift, h], axis=-1)
        y_concept_next = torch.sigmoid(net.out_concept_next(h_next))
        y_concept_all = torch.sigmoid(net.out_concept_all(h))
        outputs["y_concept_next" + add_name] = net.get_avg_fusion_concepts(
            y_concept_next, data["cshft"])
        outputs["y_concept_all" + add_name] = net.get_avg_fusion_concepts(
            y_concept_all, data["cshft"])
    return outputs


class QIKTNet(nn.Module):
    def __init__(self, num_q, num_c, emb_size, dropout=0.4,
                 mlp_layer_num=2, device="cpu", other_config=None):
        super().__init__()
        self.model_name = "qikt"
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.mlp_layer_num = mlp_layer_num
        self.device = device
        self.other_config = other_config or {}
        self.output_mode = self.other_config.get("output_mode", "an")
        self.que_emb = QueEmbIEKT(num_q=num_q, num_c=num_c,
                                  emb_size=emb_size, device=device)
        self.que_lstm_layer = nn.LSTM(self.emb_size * 4, self.hidden_size,
                                      batch_first=True)
        self.concept_lstm_layer = nn.LSTM(self.emb_size * 2,
                                          self.hidden_size,
                                          batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.out_question_next = MLP(mlp_layer_num, self.hidden_size * 3,
                                     1, dropout)
        self.out_question_all = MLP(mlp_layer_num, self.hidden_size,
                                    num_q, dropout)
        self.out_concept_next = MLP(mlp_layer_num, self.hidden_size * 3,
                                    num_c, dropout)
        self.out_concept_all = MLP(mlp_layer_num, self.hidden_size,
                                   num_c, dropout)
        self.que_disc = MLP(mlp_layer_num, self.hidden_size * 2, 1,
                            dropout)

    def get_avg_fusion_concepts(self, y_concept, cshft):
        max_num_concept = cshft.shape[-1]
        concept_mask = torch.where(cshft.long() == -1, False, True)
        concept_index = F.one_hot(torch.where(cshft != -1, cshft, 0),
                                  self.num_c)
        concept_sum = (y_concept.unsqueeze(2).repeat(
            1, 1, max_num_concept, 1) * concept_index).sum(-1)
        concept_sum = concept_sum * concept_mask
        return concept_sum.sum(-1) / torch.where(
            concept_mask.sum(-1) != 0, concept_mask.sum(-1), 1)

    def forward(self, q, c, r, data=None):
        _, emb_qca, emb_qc, emb_q, emb_c = self.que_emb(q, c, r)
        emb_qc_shift = emb_qc[:, 1:, :]
        emb_qca_current = emb_qca[:, :-1, :]
        que_h = self.dropout_layer(self.que_lstm_layer(emb_qca_current)[0])
        outputs = get_outputs(self, emb_qc_shift, que_h, data,
                              model_type="question")
        emb_ca = torch.cat(
            [emb_c.mul((1 - r).unsqueeze(-1).repeat(1, 1, self.emb_size)),
             emb_c.mul(r.unsqueeze(-1).repeat(1, 1, self.emb_size))],
            dim=-1)
        emb_ca_current = emb_ca[:, :-1, :]
        concept_h = self.dropout_layer(
            self.concept_lstm_layer(emb_ca_current)[0])
        concept_outputs = get_outputs(self, emb_qc_shift, concept_h, data,
                                      model_type="concept")
        outputs["y_concept_all"] = concept_outputs["y_concept_all"]
        outputs["y_concept_next"] = concept_outputs["y_concept_next"]
        return outputs

    def fused_y(self, outputs):
        """predict_one_step's 'an' fusion with the official output
        lambdas (q_all 1, c_all 1, c_next 1, q_next 0)."""
        oc = self.other_config
        l_c_all = oc.get("output_c_all_lambda", 1)
        l_c_next = oc.get("output_c_next_lambda", 1)
        l_q_all = oc.get("output_q_all_lambda", 1)
        y = (outputs["y_question_all"] * l_q_all
             + outputs["y_concept_all"] * l_c_all
             + outputs["y_concept_next"] * l_c_next)
        return y / (l_q_all + l_c_all + l_c_next)
