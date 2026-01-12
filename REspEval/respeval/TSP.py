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

"""
from __future__ import annotations
from typing import List, Dict, Optional, Sequence
import math
import numpy as np
from .utils_evaluate_response import response_label_classes
# --------------------------- Class mapping ---------------------------

RESPONSE_LABEL_CLASSES = response_label_classes
CLASSES: List[str] = list(RESPONSE_LABEL_CLASSES.keys())  # ["Cooperative", "Defensive", "Hedge", "Social", "NonArg"]
IDX = {c: i for i, c in enumerate(CLASSES)}

# Build fine label -> class map
LABEL2CLASS: Dict[str, str] = {}
for c, labs in RESPONSE_LABEL_CLASSES.items():
    for lab in labs:
        LABEL2CLASS[lab] = c


def _soft_vector_from_labels(labels: Sequence[str]) -> np.ndarray:
    """Map a set of fine-grained labels (possibly multiple) for one sentence
    to a 5D soft class vector over CLASSES. If multiple labels map to the same
    class, we count that class once (set semantics) to avoid double counting.
    The vector sums to 1 if at least one known label appears; otherwise all zeros.
    """
    present_classes = []
    for lab in labels:
        c = LABEL2CLASS.get(lab)
        if c is not None and c not in present_classes:
            present_classes.append(c)
    v = np.zeros(len(CLASSES), dtype=float)
    if not present_classes:
        return v
    w = 1.0 / len(present_classes)
    for c in present_classes:
        v[IDX[c]] += w
    return v


def _aggregate_shares(V: np.ndarray, weights: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Compute class shares and composite metrics from per-sentence soft vectors V (n x 5).
    If weights is None, uniform sentence-weighting is used.
    """
    n = V.shape[0]
    if n == 0:
        zeros = {c: 0.0 for c in CLASSES}
        zeros.update({"Polarity": 0.0, "ArgLoad": 0.0, "HedgeIntensity": 0.0, "Entropy": 0.0})
        return zeros
    if weights is None:
        w = np.ones((n, 1), dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1, 1)
        if w.shape[0] != n:
            raise ValueError("weights length must match number of sentences")
    totals = (V * w).sum(axis=0)
    Z = float(w.sum()) if float(w.sum()) > 0 else 1.0
    shares = totals / Z

    coop, defe, hedge, social, nonarg = shares.tolist()
    polarity = coop - defe
    argload = 1.0 - (social + nonarg)
    denom = coop + defe + hedge
    hedge_intensity = (hedge / denom) if denom > 0 else 0.0
    eps = 1e-12
    entropy = float(-(shares * np.log(shares + eps)).sum() / math.log(len(CLASSES)))

    out = {c: float(shares[IDX[c]]) for c in CLASSES}
    out.update({
        "Polarity": float(polarity),
        "ArgLoad": float(argload),
        "HedgeIntensity": float(hedge_intensity),
        "Entropy": float(entropy),
    })
    return out


def tone_stance_profile_multilabel(
    sentence_labels: List[Sequence[str]],
    word_lengths: Optional[List[int]] = None,
    *,
    cap_words: Optional[int] = 120,
    sublinear: Optional[str] = None,  # one of {None, "sqrt", "log"}
) -> Dict[str, Dict[str, float]]:
    """Compute BOTH sentence-weighted and word-weighted profiles.

    Parameters
    ----------
    sentence_labels : list[list[str]]
        Per-sentence fine-grained label sets (multi-label allowed).
    word_lengths : list[int] or None
        Word counts per sentence. If None, word-weighted == sentence-weighted.
    cap_words : int or None
        If set, caps each sentence's word contribution at this value.
    sublinear : None | "sqrt" | "log"
        Optional sublinear scaling for word weights.

    Returns
    -------
    {
      "sentence_weighted": {shares+composites},
      "word_weighted":    {shares+composites}
    }
    """
    n = len(sentence_labels)
    if n == 0:
        zeros = {c: 0.0 for c in CLASSES}
        zeros.update({"Polarity": 0.0, "ArgLoad": 0.0, "HedgeIntensity": 0.0, "Entropy": 0.0})
        return {"sentence_weighted": zeros, "word_weighted": zeros}

    # Build per-sentence soft vectors
    V = np.vstack([_soft_vector_from_labels(lbls) for lbls in sentence_labels])  # (n,5)

    # Sentence-weighted (uniform)
    sent_profile = _aggregate_shares(V, weights=None)

    # Word-weighted (with cap/sublinear if provided)
    if word_lengths is None:
        word_profile = sent_profile
    else:
        t = np.asarray(word_lengths, dtype=float)
        if cap_words is not None:
            t = np.minimum(t, float(cap_words))
        if sublinear == "sqrt":
            t = np.sqrt(t)
        elif sublinear == "log":
            t = np.log1p(t)
        word_profile = _aggregate_shares(V, weights=t)

    return {"sentence_weighted": sent_profile, "word_weighted": word_profile}



# --------------------------- Example run ---------------------------
if __name__ == "__main__":
    sent_labels = [
        ["accept for future work", "mitigate criticism"],
        ["mitigate criticism", "contradict assertion"],
        ["social", "task will be done in next version"],
    ]
    word_lengths = [22, 18, 30]

    profiles = tone_stance_profile_multilabel(sent_labels, word_lengths, cap_words=200, sublinear=None)
    print("Sentence-weighted:", profiles["sentence_weighted"]) 
    print("Word-weighted:", profiles["word_weighted"]) 

  
