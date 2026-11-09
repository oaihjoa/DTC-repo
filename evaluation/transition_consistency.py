"""HTCR / STCR transition consistency metrics.

From: Bai et al. "Beyond Next-Response Prediction: Evaluating Knowledge State
Transition Consistency in Deep Learning Based KT Models." AIED 2026, pp. 509-523.

For consecutive practices of the same KC at steps t < t' (gap <= max_gap over
the KC-expanded sequence), delta = p_hat[t'] - p_hat[t] must align with the
response r at the EARLIER step t:
  hard (Eq. 1): r==1 -> delta > 0;  r==0 -> delta < 0
  soft (Eq. 3): r==1 -> delta >= 0; r==0 -> delta <= 0
HTCR/STCR (Eq. 2) = consistent pairs / all pairs, pooled over students and KCs.

epsilon (Sect. 4.2, Obs. 4): pairs with |delta| <= epsilon are treated as
"unchanged" — consistent under soft, but EXCLUDED from HTCR's numerator and
denominator ("filters out transitions with small estimated changes").  At the
default epsilon = 0 unchanged pairs stay in the hard denominator as failures,
which is what makes STCR - HTCR equal the proportion of unchanged pairs.
Paper defaults: max_gap = 200, epsilon = 0.
"""

from dataclasses import dataclass


def iter_pairs(concepts, max_gap=200):
    """
    concept: (t,k) sequence, t being timestep, and k being knowlege
    Yield (t_prev, t_next) for consecutive same-KC practices of one student."""
    last_seen = {}
    for t, k in enumerate(concepts):
        if k in last_seen and t - last_seen[k] <= max_gap:
            yield last_seen[k], t
        last_seen[k] = t


@dataclass
class TCRResult:
    htcr: float
    stcr: float
    n_pairs: int        # all pairs (soft denominator)
    n_pairs_hard: int   # hard denominator after epsilon filtering
    breakdown: dict     # breakdown[r]["up" | "unchanged" | "down"] pair counts


def evaluate_transitions(seqs, max_gap=200, epsilon=0.0):
    """seqs: iterable of (concepts, preds, responses) per student, 1-D sequences
    of equal length (valid steps only): KC id, model P(correct), 0/1 response."""
    hard_num = hard_den = soft_num = n_pairs = 0
    breakdown = {1: {"up": 0, "unchanged": 0, "down": 0},
                 0: {"up": 0, "unchanged": 0, "down": 0}}

    for concepts, preds, responses in seqs:
        for tp, tn in iter_pairs(concepts, max_gap):
            r = int(responses[tp])
            delta = float(preds[tn]) - float(preds[tp])
            unchanged = abs(delta) <= epsilon

            n_pairs += 1
            breakdown[r]["unchanged" if unchanged else "up" if delta > 0 else "down"] += 1
            soft_num += unchanged or (r == 1) == (delta > 0) #(r = 1 and delta > 0) or (r = 0 and delta < 0)
            if not (unchanged and epsilon > 0):     # eps > 0 filters unchanged out
                hard_den += 1
                hard_num += not unchanged and (r == 1) == (delta > 0) #same here, but ignore unchanged.

    return TCRResult(
        htcr=hard_num / hard_den if hard_den else float("nan"),
        stcr=soft_num / n_pairs if n_pairs else float("nan"),
        n_pairs=n_pairs, n_pairs_hard=hard_den, breakdown=breakdown,
    )
