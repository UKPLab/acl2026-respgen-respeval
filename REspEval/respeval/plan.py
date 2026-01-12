from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np

# ----------------- Soft ROUGE L -----------------
def soft_lcs_len(
    gold_steps: List[List[str]], 
    actual_steps: List[List[str]], 
    tau: float = 0.5
) -> int:
    """
    Length of soft LCS where step equality is replaced by Jaccard >= tau.
    gold_steps / actual_steps are lists of canonicalized label-sets (lists of strings).
    """
    m, n = len(gold_steps), len(actual_steps)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            # soft match via Jaccard >= tau
            inter = len(set(gold_steps[i]) & set(actual_steps[j]))
            union = len(set(gold_steps[i]) | set(actual_steps[j]))
            jac = (inter / union) if union > 0 else 1.0  # both empty -> perfect
            if jac >= tau:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]

def soft_rougeL_on_steps(
    gold_steps: List[List[str]],
    actual_steps: List[List[str]],
    tau: float = 0.5
) -> Dict[str, float]:
    """
    Soft ROUGE-L computed on sequences of steps (lists of label-sets).
    Returns recall, precision, and F.
    """
    L = soft_lcs_len(gold_steps, actual_steps, tau=tau)
    r = L / max(1, len(gold_steps))
    p = L / max(1, len(actual_steps))
    f = 0.0 if (r + p) == 0 else 2 * r * p / (r + p)
    return {"soft_rougeL_recall": r, "soft_rougeL_precision": p, "soft_rougeL_f": f}


# ----------------- Helpers -----------------

def canon_label(s: str) -> str:
    """Canonicalize a label string."""
    return " ".join(s.strip().lower().split())

def canon_step(step: List[str]) -> List[str]:
    """Canonicalize a multi-label step: sort unique labels."""
    return sorted({canon_label(x) for x in step if x and canon_label(x)})

def extract_items(plan: Dict[str, Any], cat: str) -> List[Dict[str, Any]]:
    """Return list of items for category, or [] if missing."""
    return plan.get(cat, []) or []

def steps_of(item: Dict[str, Any]) -> List[List[str]]:
    """Get list of steps (list of labels) for an item, canonicalized."""
    raw = item.get("response_plan", []) or []
    return [canon_step(step) for step in raw]

def bag_of_labels(steps: List[List[str]]) -> Counter:
    """Multiset of labels across steps (counts duplicates across steps)."""
    c = Counter()
    for st in steps:
        c.update(st)
    return c

def multiset_prf(pred: Counter, gold: Counter) -> Dict[str, float]:
    inter = sum((pred & gold).values())
    p = inter / max(1, sum(pred.values()))
    r = inter / max(1, sum(gold.values()))
    f = 0.0 if p+r == 0 else 2*p*r/(p+r)
    return {"precision": p, "recall": r, "f1": f}

def lcs_len(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            dp[i+1][j+1] = dp[i][j]+1 if a[i]==b[j] else max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]

def tokenize_steps(steps: List[List[str]]) -> List[str]:
    """Turn each multi-label step into a canonical token (order-aware on steps, order-agnostic within a step)."""
    return ["+".join(step) if step else "" for step in steps]

def jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B: return 1.0
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def greedy_step_match(gold_steps: List[List[str]], pred_steps: List[List[str]], thresh: float=0.5) -> Tuple[List[int], float]:
    """
    Greedily match each gold step to at most one pred step by maximum Jaccard.
    Returns matched pred indices (or -1) and mean Jaccard over matched gold steps.
    """
    matched = [-1]*len(gold_steps)
    used = set()
    sims = []
    for i, g in enumerate(gold_steps):
        best_j, best_j_idx = -1.0, -1
        for j, p in enumerate(pred_steps):
            if j in used:
                continue
            jac = jaccard(g, p)
            if jac > best_j:
                best_j, best_j_idx = jac, j
        if best_j >= thresh:
            matched[i] = best_j_idx
            used.add(best_j_idx)
        sims.append(max(best_j, 0.0))
    mean_j = sum(sims)/len(sims) if sims else 0.0
    return matched, mean_j

def order_score_from_matches(matched_indices: List[int]) -> float:
    """Compute LCS over the sequence of matched indices ignoring -1, normalized by number of matched."""
    seq = [idx for idx in matched_indices if idx >= 0]
    if not seq:
        return 0.0
    # Compare to their sorted order; order fidelity = LCS(seq, sorted(seq))/len(seq)
    ideal = sorted(seq)
    l = lcs_len(seq, ideal)
    return l / len(seq)

def rougeL_on_step_tokens(pred_steps: List[List[str]], gold_steps: List[List[str]]) -> Dict[str,float]:
    p_tok = tokenize_steps(pred_steps)
    g_tok = tokenize_steps(gold_steps)
    l = lcs_len(p_tok, g_tok)
    r = l / max(1, len(g_tok))
    p = l / max(1, len(p_tok))
    f = 0.0 if (p+r)==0 else 2*p*r/(p+r)
    return {"rougeL_recall": r, "rougeL_precision": p, "rougeL_f": f}

def plan_fulfilled_set(plan_steps: List[List[str]], exec_steps: List[List[str]]) -> int:
    """Binary: all plan labels appear somewhere in execution (multiset subset)."""
    return int(all((bag_of_labels(exec_steps)[lab] >= cnt) for lab, cnt in bag_of_labels(plan_steps).items()))

def plan_is_subsequence(plan_steps: List[List[str]], exec_steps: List[List[str]], thresh: float=0.8) -> int:
    """
    Binary: is plan a subsequence of exec where a plan step is 'matched' if Jaccard>=thresh?
    """
    j = 0
    for p in plan_steps:
        found = False
        while j < len(exec_steps):
            if jaccard(p, exec_steps[j]) >= thresh:
                found = True
                j += 1
                break
            j += 1
        if not found:
            return 0
    return 1

# --------------- Core comparisons per item ----------------

def compare_item(pred_steps: List[List[str]], gold_steps: List[List[str]]) -> Dict[str, float]:
    # 1) Label bag PRF
    lbl = multiset_prf(bag_of_labels(pred_steps), bag_of_labels(gold_steps))

    # 2) Step coverage & order via greedy Jaccard matching
    matched, mean_j = greedy_step_match(gold_steps, pred_steps, thresh=0.5)
    step_recall = sum(1 for m in matched if m >= 0) / max(1, len(gold_steps))
    order_fidelity = order_score_from_matches(matched)

    # 3) Order-aware sequence token ROUGE-L
    rl = rougeL_on_step_tokens(pred_steps, gold_steps)

    srl = soft_rougeL_on_steps(gold_steps, pred_steps, tau=0.5)

    return {
        "label_precision": lbl["precision"],
        "label_recall": lbl["recall"],
        "label_f1": lbl["f1"],
        "step_recall": step_recall,
        "mean_step_jaccard": mean_j,
        "order_fidelity": order_fidelity,
        "rougeL_recall": rl["rougeL_recall"],
        "rougeL_precision": rl["rougeL_precision"],
        "rougeL_f": rl["rougeL_f"],
        "soft_rougeL_recall": srl["soft_rougeL_recall"],
        "soft_rougeL_precision": srl["soft_rougeL_precision"],
        "soft_rougeL_f": srl["soft_rougeL_f"],
    }

def fulfillment_metrics(plan_steps: List[List[str]], exec_steps: List[List[str]]) -> Dict[str, float]:
    return {
        "plan_fulfilled_set": float(plan_fulfilled_set(plan_steps, exec_steps)),
        "plan_fulfilled_subsequence": float(plan_is_subsequence(plan_steps, exec_steps, thresh=0.8)),
    }

# --------------- End-to-end driver ----------------

def evaluate_plan_controllability(gold: Dict[str,Any], plan: Dict[str,Any], actual: Dict[str,Any]) -> Dict[str,Any]:
    """
    Compare: planning accuracy (plan vs gold), execution fidelity (actual vs plan),
    and end-to-end adherence (actual vs gold).
    Returns a nested report with macro-averages.
    """
    cats = ["questions", "criticisms", "requests"]
    report = {"per_item": [], "macro": {}}

    # Flatten all items with (cat, id)
    items = []
    for cat in cats:
        Gs = extract_items(gold, cat)
        Ps = extract_items(plan, cat)
        Es = extract_items(actual, cat)
        # Assume same order / ids; if mismatch, align by id key if present
        # Build dicts by id when available
        def by_id(lst):
            if lst and "id" in lst[0]:
                return {x["id"]: x for x in lst}
            return {i: x for i, x in enumerate(lst)}
        Gd, Pd, Ed = by_id(Gs), by_id(Ps), by_id(Es)
        ids = sorted(set(Gd) | set(Pd) | set(Ed))
        for iid in ids:
            items.append((cat, iid, Gd.get(iid), Pd.get(iid), Ed.get(iid)))

    # Collect metrics per item
    accum = {
        "plan_vs_gold": [],
        "actual_vs_plan": [],
        "actual_vs_gold": [],
        "fulfillment": [],
    }

    for cat, iid, Gi, Pi, Ei in items:
        
        g_steps = steps_of(Gi) if Gi else []
        p_steps = steps_of(Pi) if Pi else []
        e_steps = steps_of(Ei) if Ei else []

        m_pg = compare_item(p_steps, g_steps)
        m_ap = compare_item(e_steps, p_steps)
        m_ag = compare_item(e_steps, g_steps)
        m_f  = fulfillment_metrics(p_steps, e_steps)

        report["per_item"].append({
            "category": cat,
            "id": iid,
            "sizes": {"gold_steps": len(g_steps), "plan_steps": len(p_steps), "actual_steps": len(e_steps)},
            "plan_vs_gold": m_pg,
            "actual_vs_plan": m_ap,
            "actual_vs_gold": m_ag,
            "fulfillment": m_f,
        })

        accum["plan_vs_gold"].append(m_pg)
        accum["actual_vs_plan"].append(m_ap)
        accum["actual_vs_gold"].append(m_ag)
        accum["fulfillment"].append(m_f)

    # Macro-averages
    def avg_dict(lst: List[Dict[str,float]]) -> Dict[str,float]:
        if not lst: return {}
        keys = lst[0].keys()
        return {k: sum(d.get(k,0.0) for d in lst)/len(lst) for k in keys}

    report["macro"]["plan_vs_gold"]   = avg_dict(accum["plan_vs_gold"])
    report["macro"]["actual_vs_plan"] = avg_dict(accum["actual_vs_plan"])
    report["macro"]["actual_vs_gold"] = avg_dict(accum["actual_vs_gold"])
    report["macro"]["fulfillment"]    = avg_dict(accum["fulfillment"])

    return report
