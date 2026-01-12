"""
Tone–Stance metrics for author responses with multi-label sentence annotations.
And interaction flow / trajectory metrics.

Tone–Stance Metric definitions
------------------
- Cooperative (Coop): proportion of text that answers questions, accepts future work, concedes, or reports completed/will-do tasks.
- Defensive (Defe): proportion of text that rejects, refutes, or contradicts reviewer points.
- Hedge: proportion of text that downplays or mitigates reviewer concerns.
- Social: proportion of text that is social or praise-only (no substantive argument).
- NonArg: proportion of text used for structure, summarization, follow-up, or otherwise non-argumentative content.
- Polarity = Cooperative − Defensive. Higher → more cooperative stance, lower → more defensive.
- ArgLoad = 1 − (Social + NonArg). Proportion of content that carries substantive argumentative load.
- HedgeIntensity = Hedge / (Cooperative + Defensive + Hedge). Fraction of argumentative content that uses hedging.
- Entropy = normalized entropy over the 5-class distribution. Higher → more diverse/mixed tones; lower → more focused tone.

Interaction flow metrics
------------------------
- STM (Stance Transition Matrix): 5x5 row-stochastic matrix of transition probabilities between stance classes across consecutive sentences.
- StartPolarity: polarity value of the first sentence.
- EndPolarity: polarity value of the last sentence.
- NetShift: EndPolarity − StartPolarity. Positive → becomes more cooperative; negative → becomes more defensive.
- Slope: linear trend of polarity across sentences (positive slope means cooperativeness grows over the response).
- Volatility: standard deviation of polarity across sentences, indicating stance variability.
- CHR (Hedge-after-Commit Rate): probability that a Cooperative sentence is followed by a Hedge.
- CDR (Defensive Escalation Rate): probability that a Cooperative sentence is followed by a Defensive one.
- DC/HR (Softening Rate): probability that a Defensive sentence is followed by Cooperative or Hedge, i.e., softening after defense.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Sequence
import math
import numpy as np
from REspEval.respeval.TSP import _soft_vector_from_labels, RESPONSE_LABEL_CLASSES, CLASSES, IDX, LABEL2CLASS

# --------------------------- Class Transition & Position Analysis ---------------------------

def _row_stochastic(M: np.ndarray) -> np.ndarray:
    """Row-normalize; keep all-zero rows as zeros; always return float."""
    M = np.asarray(M, dtype=float)
    rs = M.sum(axis=1, keepdims=True)
    out = np.zeros_like(M)
    np.divide(M, np.where(rs == 0.0, 1.0, rs), out=out, where=rs != 0.0)
    return out

def _build_soft_vectors(sentence_labels: List[Sequence[str]]) -> np.ndarray:
    return np.vstack([_soft_vector_from_labels(lbls) for lbls in sentence_labels]) if sentence_labels else np.zeros((0,5))

def _transition_counts(V: np.ndarray) -> np.ndarray:
    n = V.shape[0]
    T = np.zeros((5,5), dtype=float)
    for t in range(n-1):
        T += np.outer(V[t], V[t+1])
    return T

def _positional_hist(V: np.ndarray, bins: int = 10) -> Dict[str, List[float]]:
    n = V.shape[0]
    edges = np.linspace(0.0, 1.0, bins + 1)
    rel_pos = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.0])
    H = {c: np.zeros(bins, dtype=float) for c in CLASSES}
    for i in range(n):
        b = min(bins - 1, int(np.searchsorted(edges, rel_pos[i], side='right') - 1))
        for ci, c in enumerate(CLASSES):
            H[c][b] += V[i, ci]
    return {c: H[c].tolist() for c in CLASSES}


def class_transition_position_analysis(
    sentence_labels: List[Sequence[str]],
    *,
    bins: int = 5,
) -> Dict[str, object]:
    """Richer flow analysis over ALL classes (not just polarity).

    Answers:
      • Do responses tend to start with Social or Cooperative?  -> start_probs
      • How do classes distribute by relative position?         -> pos_density (per class over bins)
      • Are there Hedge-after-Coop or Coop-after-Hedge patterns?-> pairwise STM entries
      • Where are Defensive sentences and what surrounds them?   -> ctx_prev/ctx_next for each class (esp. Defensive)

    Returns
    -------
    {
      'start_probs': {class: p},                      # soft class dist. of first sentence
      'end_probs':   {class: p},                      # last sentence
      'mean_position': {class: mean_pos_in_[0,1]},    # expected relative position per class
      'pos_density': {class: [p_bin0,...,p_bin{bins-1}]},   # normalized per class over bins
      'STM': 5x5 row-stochastic list (same as interaction_flow_metrics),
      'pairwise': {(c1,c2): P(next=c2|curr=c1)},
      'context_prev': {class: {prev_class: P(prev|curr)}},
      'context_next': {class: {next_class: P(next|curr)}},
    }
    """
    n = len(sentence_labels)
    if n == 0:
        zero_vec = {c: 0.0 for c in CLASSES}
        return {
            "StartPolarity": 0.0,
            "EndPolarity": 0.0,
            "NetShift": 0.0,
            "Slope": 0.0,
            "Volatility": 0.0,
            'start_probs': zero_vec,
            'end_probs': zero_vec,
            'mean_position': zero_vec,
            'pos_density': {c: [0.0]*bins for c in CLASSES},
            'STM': [[0]*5 for _ in range(5)],
            'pairwise': {},
            'context_prev': {c: zero_vec for c in CLASSES},
            'context_next': {c: zero_vec for c in CLASSES},
        }

    V = _build_soft_vectors(sentence_labels)

    # Polarity per sentence (uniform per sentence)
    pol = (V[:, IDX["Cooperative"]] - V[:, IDX["Defensive"]]).astype(float)
    start_p, end_p = float(pol[0]), float(pol[-1])
    net_shift = float(end_p - start_p)

    # slope via least squares
    if n >= 2:
        x = np.arange(n, dtype=float)
        X = np.vstack([x, np.ones_like(x)]).T
        slope, _ = np.linalg.lstsq(X, pol, rcond=None)[0]
        # soft transition counts
        T = np.zeros((5, 5), dtype=float)
        for t in range(n - 1):
            T += np.outer(V[t], V[t + 1])
        row_sums = T.sum(axis=1, keepdims=True)
        STM = np.divide(T, np.where(row_sums == 0, 1.0, row_sums), where=row_sums != 0)
    else:
        slope = 0.0
        STM = np.zeros((5, 5), dtype=float)
    volatility = float(np.std(pol))

    # Start/End distributions (soft)
    start_probs = {c: float(V[0, IDX[c]]) for c in CLASSES}
    end_probs   = {c: float(V[-1, IDX[c]]) for c in CLASSES}

    # Relative position in [0,1]
    rel_pos = np.linspace(0.0, 1.0, V.shape[0]) if V.shape[0] > 1 else np.array([0.0])

    # Mean position per class
    mean_position = {}
    for ci, c in enumerate(CLASSES):
        w = V[:, ci]
        Z = float(w.sum()) or 1.0
        mean_position[c] = float((w * rel_pos).sum() / Z)

    # Positional density by bins per class
    pos_density = _positional_hist(V, bins=bins)

    # Transition matrix over all classes
    T = _transition_counts(V)
    STM = _row_stochastic(T)

    # Pairwise dict
    pairwise = {}
    for i, c1 in enumerate(CLASSES):
        row = STM[i]
        denom = row.sum() or 1.0
        for j, c2 in enumerate(CLASSES):
            pairwise[f'{c1}=>{c2}'] = float(row[j] / denom)

    # Context distributions: P(prev | curr=c) and P(next | curr=c)
    ctx_prev = {c: {k: 0.0 for k in CLASSES} for c in CLASSES}
    ctx_next = {c: {k: 0.0 for k in CLASSES} for c in CLASSES}

    for t in range(V.shape[0]):
        for ci, c in enumerate(CLASSES):
            wt = V[t, ci]
            if wt == 0:
                continue
            if t-1 >= 0:
                for kj, k in enumerate(CLASSES):
                    ctx_prev[c][k] += wt * V[t-1, kj]
            if t+1 < V.shape[0]:
                for kj, k in enumerate(CLASSES):
                    ctx_next[c][k] += wt * V[t+1, kj]

    # Normalize per current class
    for c in CLASSES:
        s = sum(ctx_prev[c].values()) or 1.0
        for k in CLASSES:
            ctx_prev[c][k] = float(ctx_prev[c][k] / s)
        s = sum(ctx_next[c].values()) or 1.0
        for k in CLASSES:
            ctx_next[c][k] = float(ctx_next[c][k] / s)

    return {
        "StartPolarity": start_p,
        "EndPolarity": end_p,
        "NetShift": net_shift,
        "Slope": float(slope),
        "Volatility": volatility,
        'start_probs': start_probs,
        'end_probs': end_probs,
        'mean_position': mean_position,
        'pos_density': pos_density,
        'STM': STM.tolist(),
        'pairwise': pairwise,
        'context_prev': ctx_prev,
        'context_next': ctx_next,
    }

# --------------------------- Example run ---------------------------
if __name__ == "__main__":
    sent_labels = [
        ["accept for future work", "mitigate criticism"],
        ["mitigate criticism", "contradict assertion"],
        ["social", "task will be done in next version"],
    ]
    word_lengths = [22, 18, 30]

    flow = class_transition_position_analysis(sent_labels, bins=5)

  
