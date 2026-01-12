"""
Utilities for aggregating Tone–Stance Profiles (TSP) and full-class
Transition/Position flow across many samples, with CSV exporters
and simple matplotlib plotters.

Note: "samples" refer to analysis units (e.g., one review item), *not* documents.
Each sample: {"labels": List[List[str]], "lengths": Optional[List[int]]}

Exports:
- aggregate_over_samples(samples, bins=10)
- bootstrap_aggregate(samples, n=1000, bins=10, random_state=None)
- export_profiles_to_csv(agg, csv_path)
- export_pairwise_to_csv(agg_or_boot, csv_path)
- export_position_to_csv(agg, csv_path)
- plot_stm_heatmap(stm, out_path=None)
- plot_positional_density(pos_density, out_path=None)
- plot_shares_bar(shares, title="Shares", out_path=None)
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple, Any
from pathlib import Path
import numpy as np
import csv
import matplotlib.pyplot as plt
import os

from REspEval.respeval.TSP import CLASSES
from REspEval.respeval.flow import _row_stochastic
    
# --------------------------- Metric Keys ---------------------------
# canonical order (classes + composites)
TSP_KEYS = [
    "Cooperative", "Defensive", "Hedge", "Social", "NonArg",
    "Polarity", "ArgLoad", "HedgeIntensity", "Entropy",
]

def aggregate_and_plot_tsp_flow(path, gen_type, outdir):
    # -------- TSP and flow aggregation and plots -------
    _outdir = Path(outdir)/ 'TSP_flow'
    _outdir.mkdir(parents=True, exist_ok=True)

    # Aggregate (macro over saved)
    agg_tsp  = aggregate_TSP_from_saved(path, gen_type)
    agg_flow = aggregate_flow_from_saved(path, gen_type)
    agg = {**agg_tsp, **agg_flow}
    _file = _outdir / "TSP-flow_agg.json"
    _file = Path(_handle_windows_path_length_limit(_file))
    save_json(agg, _file)

    # Bootstrap CIs
    boot_file = _outdir / "TSP-flow_bootstrap.json"
    boot_file = Path(_handle_windows_path_length_limit(boot_file))
    if Path(boot_file).exists():
        boot = load_json(boot_file)
    else:
        boot = bootstrap_aggregate_from_saved(path, gen_type, n=1000, alpha=0.05, random_state=42)
        save_json(boot, boot_file)

    # --- TSP exports/plots
    export_TSP_to_csv(agg, str(_outdir / "agg_TSP.csv"), boot=boot)
    #plot_TSP_class_bar(agg["TSP"]["word_weighted"], "", str(_outdir / "TSP_word.png"))
    
    # --- Flow exports/plots
    export_pairwise_to_csv(agg, str(_outdir / "agg_flow_pairwise.csv"), boot=boot)
    export_flow_scalars_to_csv(agg, str(_outdir / "agg_flow_scalars.csv"), boot=boot)
    export_start_end_to_csv(agg, str(_outdir / "agg_flow_start_end.csv"), boot=boot)
    export_position_to_csv(agg, str(_outdir / "agg_flow_position.csv"))  # no CI for densities

    stm = np.asarray(agg["flow"]["STM"], dtype=float)
    plot_stm_heatmap(stm, str(_outdir / "flow_stm.png"))
    #plot_positional_density(agg["flow"]["pos_density"], str(_outdir / "flow_pos_density.png"))
    
    #plot_start_end_probs(agg["flow"], str(_outdir / "flow_start_end_probs.png"))
    
    
    return agg



# --------------------------- JSON I/O ---------------------------
def load_json(fp: Path) -> Dict:
    with open(fp, "r", encoding="utf-8") as f:
       return json.load(f)


def save_json(obj: Dict, fp: Path) -> None:
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)

# --------------------------- Aggregation over samples ---------------------------

### TSP aggregation

def macro_mean(dicts: List[Dict[str, float]], keys: List[str]) -> Dict[str, float]:
    """Average keys across a list of dicts (macro). Missing keys are skipped per-key."""
    out = {}
    for k in keys:
        vals = [d.get(k) for d in dicts if d.get(k) is not None]
        if vals:
            out[k] = float(np.mean(vals))

    return out


def aggregate_TSP_from_saved(path: str, gen_type:str, sample_subset: Optional[List[Path]] = None) -> Dict[str, object]:
    """Aggregate macro TSP over a list of per-sample JSON files.

    Each file should contain at least one of the keys:
      - "TSP_sentence_weighted"
      - "TSP_word_weighted"

    Returns a dict with macro means.
    """
    sent_dicts, word_dicts = [], []
    base = Path(path)
    samples = sample_subset or [
        p for p in base.iterdir() if p.is_dir() and not p.name.startswith("agg_")
    ]

    for p in samples:
            subfolder = p / gen_type / 'TSP_flow'
            file = subfolder / f'gpt-5_output_scores_TSP.json'
            file = Path(_handle_windows_path_length_limit(file))
            obj = load_json(file)
        
            sw = obj.get("TSP_sentence_weighted")
            ww = obj.get("TSP_word_weighted")
            if isinstance(sw, dict):
                sent_dicts.append(sw)
            if isinstance(ww, dict):
                word_dicts.append(ww)

    result = {
        "TSP": {
            "sentence_weighted": macro_mean(sent_dicts, TSP_KEYS) if sent_dicts else {},
            "word_weighted": macro_mean(word_dicts, TSP_KEYS) if word_dicts else {},
        },
    }
    return result

### Flow aggregation

# ---------- math helpers ----------

def _avg_scalar(flow_jsons: List[Dict[str, Any]], key: str) -> float:
    return float(np.mean([float(f.get(key, 0.0)) for f in flow_jsons]))

def _avg_class_dicts(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    return {c: float(np.mean([d.get(c, 0.0) for d in dicts])) for c in CLASSES}

def _avg_pairwise(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    keys = [f"{a}=>{b}" for a in CLASSES for b in CLASSES]
    return {k: float(np.mean([d.get(k, 0.0) for d in dicts])) for k in keys}

# ---------- sanity checks ----------
def _check_nonnegative(x: np.ndarray, name: str, atol: float) -> None:
    if np.any(x < -atol):
        raise ValueError(f"{name} has negative entries (min={float(x.min())}).")

def _check_prob_vector(v: np.ndarray, name: str, atol: float = 1e-6) -> None:
    v = np.asarray(v, dtype=float)
    _check_nonnegative(v, name, atol)
    if np.allclose(v, 0.0, atol=atol):
        return
    s = float(v.sum())
    if not np.isclose(s, 1.0, atol=atol):
        raise ValueError(f"{name} sums to {s:.8f}, expected 1.0.")

def _check_row_stochastic(M: np.ndarray, name: str, atol: float = 1e-6) -> None:
    M = np.asarray(M, dtype=float)
    _check_nonnegative(M, name, atol)
    for i, row in enumerate(M):
        _check_prob_vector(row, f"{name}[row={i}]", atol)

# ---------- core aggregator (always renormalize) ----------
N = len(CLASSES)
def _aggregate_flow(flow_jsons: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not flow_jsons:
        return {
            "StartPolarity": 0.0, "EndPolarity": 0.0, "NetShift": 0.0, "Slope": 0.0, "Volatility": 0.0,
            "start_probs": {c: 0.0 for c in CLASSES},
            "end_probs":   {c: 0.0 for c in CLASSES},
            "mean_position": {c: 0.0 for c in CLASSES},
            "pos_density": {c: [] for c in CLASSES},
            "STM": [[0.0]*N for _ in range(N)],
            "pairwise": {f"{a}=>{b}": 0.0 for a in CLASSES for b in CLASSES},
            "context_prev": {c: {k: 0.0 for k in CLASSES} for c in CLASSES},
            "context_next": {c: {k: 0.0 for k in CLASSES} for c in CLASSES},
        }

    # scalars
    StartPolarity = _avg_scalar(flow_jsons, "StartPolarity")
    EndPolarity   = _avg_scalar(flow_jsons, "EndPolarity")
    NetShift      = _avg_scalar(flow_jsons, "NetShift")
    Slope         = _avg_scalar(flow_jsons, "Slope")
    Volatility    = _avg_scalar(flow_jsons, "Volatility")

    # class dicts
    start_probs   = _avg_class_dicts([f.get("start_probs", {}) for f in flow_jsons])
    end_probs     = _avg_class_dicts([f.get("end_probs",   {}) for f in flow_jsons])
    mean_position = _avg_class_dicts([f.get("mean_position", {}) for f in flow_jsons])

    # pos_density (bins must match)
    first_pd = flow_jsons[0].get("pos_density", {})
    first_c = next((c for c in CLASSES if isinstance(first_pd.get(c), list)), None)
    bins = len(first_pd[first_c]) if first_c else 0
    for f in flow_jsons:
        d = f.get("pos_density", {})
        for c in CLASSES:
            v = d.get(c, [])
            if isinstance(v, list) and len(v) not in (0, bins):
                raise ValueError(f"pos_density bin mismatch for '{c}': expected {bins}, got {len(v)}")

    pos_density: Dict[str, List[float]] = {}
    for c in CLASSES:
        rows = [f.get("pos_density", {}).get(c, []) for f in flow_jsons]
        rows = [r for r in rows if isinstance(r, list) and len(r) == bins]
        if rows:
            m = np.mean(np.asarray(rows, dtype=float), axis=0)
            Z = float(m.sum())
            if Z > 0:
                m = m / Z
            pos_density[c] = m.tolist()
        else:
            pos_density[c] = [0.0]*bins

    # STM (macro avg then row normalize)
    STMs = [np.asarray(f.get("STM", [[0.0]*N for _ in range(N)]), dtype=float) for f in flow_jsons]
    STMs = [M for M in STMs if M.shape == (N, N)]
    STM = _row_stochastic(np.mean(np.stack(STMs, axis=0), axis=0)) if STMs else np.zeros((N, N))

    # pairwise (macro avg from inputs; NOT recomputed)
    pairwise = _avg_pairwise([f.get("pairwise", {}) for f in flow_jsons])

    # context (macro avg then row normalize)
    def _avg_context(key: str) -> Dict[str, Dict[str, float]]:
        rows = [f.get(key, {}) for f in flow_jsons]
        out = {}
        for c in CLASSES:
            mat = [[float(r.get(c, {}).get(k, 0.0)) for k in CLASSES] for r in rows]
            v = np.mean(np.asarray(mat, dtype=float), axis=0)
            Z = float(v.sum())
            if Z > 0:
                v = v / Z
            out[c] = {k: float(v[i]) for i, k in enumerate(CLASSES)}
        return out

    context_prev = _avg_context("context_prev")
    context_next = _avg_context("context_next")

    # sanity checks
    atol = 1e-6
    for c in CLASSES:
        _check_prob_vector(np.array(pos_density[c], dtype=float), f"pos_density['{c}']", atol)
    _check_row_stochastic(np.asarray(STM, dtype=float), "STM", atol)
    for c in CLASSES:
        _check_prob_vector(np.array([context_prev[c][k] for k in CLASSES], dtype=float),
                           f"context_prev['{c}']", atol)
        _check_prob_vector(np.array([context_next[c][k] for k in CLASSES], dtype=float),
                           f"context_next['{c}']", atol)

    return {
        "StartPolarity": StartPolarity,
        "EndPolarity": EndPolarity,
        "NetShift": NetShift,
        "Slope": Slope,
        "Volatility": Volatility,
        "start_probs": start_probs,
        "end_probs": end_probs,
        "mean_position": mean_position,
        "pos_density": pos_density,
        "STM": STM.tolist(),
        "pairwise": pairwise,
        "context_prev": context_prev,
        "context_next": context_next,
    }

def _handle_windows_path_length_limit(path: Path) -> str:
        # deal with potential Windows path length limitation
        # ensure parent exists (harmless if already exists)
        if isinstance(path, str):
            path = Path(path)
        #path.parent.mkdir(parents=True, exist_ok=True)

        # convert to absolute and prefix with \\?\  on Windows
        abs_path = path.resolve()
        if os.name == "nt":
            path_for_open = r"\\?\{}".format(str(abs_path))
        else:
            path_for_open = str(abs_path)
        return path_for_open

# ---------- public API ----------
def aggregate_flow_from_saved(path: str, gen_type: str, sample_subset: Optional[List[Path]] = None) -> Dict[str, Any]:
    """
    Aggregate macro 'Transition_flow' across all items in questions/criticisms/requests
    from files:
        <path>/<sample>/<gen_type>/TSP_flow/gpt-5_output_dict_with_TSP_and_flow.json

    Returns:
        {"flow": <aggregated dict>}
    """
    base = Path(path)
    samples = sample_subset or [
        p for p in base.iterdir() if p.is_dir() and not p.name.startswith("agg_")
    ]

    flows: List[Dict[str, Any]] = []
    for sample_dir in samples:
        jf = sample_dir / gen_type / "TSP_flow" /"gpt-5_output_dict_with_TSP_and_flow.json"
        jf = Path(_handle_windows_path_length_limit(jf))
        if not jf.is_file():
            continue
        try:
            obj = load_json(jf)
        except Exception:
            continue

        # Collect flows from questions / criticisms / requests
        for section in ("questions", "criticisms", "requests"):
            items = obj.get(section, [])
            if not isinstance(items, list):
                continue
            for it in items:
                # usual location
                tf = (it.get("respeval_scores", {}) or {}).get("Transition_flow")
                # fallback: directly under item (just in case)
                if tf is None:
                    tf = it.get("Transition_flow")
                if isinstance(tf, dict):
                    flows.append(tf)
    return {"flow": _aggregate_flow(flows)}



# --------------------------- Bootstrap ---------------------------

# assumes CLASSES, aggregate_TSP_from_saved, aggregate_flow_from_saved already defined

def _percentile_ci(x: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Return (mean, lo, hi) percentile CI."""
    lo = float(np.percentile(x, 100 * (alpha / 2)))
    hi = float(np.percentile(x, 100 * (1 - alpha / 2)))
    mu = float(np.mean(x))
    return mu, lo, hi

def _stats_dict(arr: List[float], alpha: float) -> Dict[str, float]:
    mu, lo, hi = _percentile_ci(np.array(arr, dtype=float), alpha)
    return {"mean": mu, "ci_lo": lo, "ci_hi": hi}

def bootstrap_aggregate_from_saved(
    path: str,
    gen_type: str,
    n: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Bootstrap CIs by resampling <path>/<sample>/ directories with replacement.
    Output mirrors your 'agg' layout: {'TSP': {...}, 'flow': {...}} but each value is a dict
    with {'mean', 'ci_lo', 'ci_hi'}.

    TSP:
        - sentence_weighted[class]
        - word_weighted[class]
    flow:
        - scalars: StartPolarity, EndPolarity, NetShift, Slope, Volatility
        - start_probs[class], end_probs[class]
        - pairwise['A=>B']
    """
    base = Path(path)
    samples = [p for p in base.iterdir() if p.is_dir() and not p.name.startswith("agg_")]
    m = len(samples)
    if m == 0:
        raise ValueError("No sample folders found under path")

    rng = np.random.default_rng(random_state)

    # storage
    # TSP
    sw = {c: [] for c in CLASSES}   # sentence-weighted
    ww = {c: [] for c in CLASSES}   # word-weighted
    # flow
    scalar_keys = ["StartPolarity", "EndPolarity", "NetShift", "Slope", "Volatility"]
    scalars = {k: [] for k in scalar_keys}
    start_probs = {c: [] for c in CLASSES}
    end_probs   = {c: [] for c in CLASSES}
    pair_keys = [f"{a}=>{b}" for a in CLASSES for b in CLASSES]
    pairwise = {k: [] for k in pair_keys}

    for _ in range(n):
        idx = rng.integers(low=0, high=m, size=m)
        subset = [samples[i] for i in idx]

        tsp_agg  = aggregate_TSP_from_saved(path, gen_type, sample_subset=subset)
        flow_agg = aggregate_flow_from_saved(path, gen_type, sample_subset=subset)

        # TSP
        sent = tsp_agg.get("TSP", {}).get("sentence_weighted", {}) or {}
        word = tsp_agg.get("TSP", {}).get("word_weighted",   {}) or {}
        for c in CLASSES:
            sw[c].append(float(sent.get(c, 0.0)))
            ww[c].append(float(word.get(c, 0.0)))

        # flow
        flow = flow_agg["flow"]
        for k in scalar_keys:
            scalars[k].append(float(flow.get(k, 0.0)))

        sp = flow.get("start_probs", {}) or {}
        ep = flow.get("end_probs",   {}) or {}
        for c in CLASSES:
            start_probs[c].append(float(sp.get(c, 0.0)))
            end_probs[c].append(float(ep.get(c, 0.0)))

        for k in pair_keys:
            pairwise[k].append(float(flow.get("pairwise", {}).get(k, 0.0)))

    # pack results to mirror agg layout
    return {
        "TSP": {
            "sentence_weighted": {c: _stats_dict(sw[c], alpha) for c in CLASSES},
            "word_weighted":     {c: _stats_dict(ww[c], alpha) for c in CLASSES},
        },
        "flow": {
            "scalars": {k: _stats_dict(scalars[k], alpha) for k in scalar_keys},
            "start_probs": {c: _stats_dict(start_probs[c], alpha) for c in CLASSES},
            "end_probs":   {c: _stats_dict(end_probs[c],   alpha) for c in CLASSES},
            "pairwise":    {k: _stats_dict(pairwise[k],    alpha) for k in pair_keys},
        },
    }

# --------------------------- Exporters ---------------------------
def export_TSP_to_csv(agg: Dict[str, Any], csv_path: str, boot: Optional[Dict[str, Any]] = None) -> None:
    """
    Save TSP class shares. If boot is provided, include mean's CI columns.
    Rows: (type, class, value, ci_lo, ci_hi) where type in {"sentence_weighted", "word_weighted"}.
    """
    tsp = agg.get("TSP", {})
    sw = tsp.get("sentence_weighted", {}) or {}
    ww = tsp.get("word_weighted",   {}) or {}

    boot_sw = (boot or {}).get("TSP", {}).get("sentence_weighted", {})
    boot_ww = (boot or {}).get("TSP", {}).get("word_weighted",   {})

    csv_path = Path(_handle_windows_path_length_limit(Path(csv_path)))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "class", "value", "ci_lo", "ci_hi"])
        for c in CLASSES:
            s = float(sw.get(c, 0.0))
            b = boot_sw.get(c) if boot_sw else None
            w.writerow(["sentence_weighted", c, s, (b or {}).get("ci_lo", ""), (b or {}).get("ci_hi", "")])
        for c in CLASSES:
            s = float(ww.get(c, 0.0))
            b = boot_ww.get(c) if boot_ww else None
            w.writerow(["word_weighted", c, s, (b or {}).get("ci_lo", ""), (b or {}).get("ci_hi", "")])


def export_pairwise_to_csv(agg: Dict[str, Any], csv_path: str, boot: Optional[Dict[str, Any]] = None) -> None:
    """
    Export aggregated pairwise transitions (long form).
    If boot provided, include ci_lo/ci_hi.
    Rows: (pair, value, ci_lo, ci_hi) where pair like "A=>B".
    """
    flow = agg.get("flow", {})
    pairwise = flow.get("pairwise", {}) or {}
    boot_pair = (boot or {}).get("flow", {}).get("pairwise", {}) or {}
    csv_path = Path(_handle_windows_path_length_limit(Path(csv_path)))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair", "value", "ci_lo", "ci_hi"])
        for pair in [f"{a}=>{b}" for a in CLASSES for b in CLASSES]:
            v = float(pairwise.get(pair, 0.0))
            b = boot_pair.get(pair)
            w.writerow([pair, v, (b or {}).get("ci_lo", ""), (b or {}).get("ci_hi", "")])


def export_flow_scalars_to_csv(agg: Dict[str, Any], csv_path: str, boot: Optional[Dict[str, Any]] = None) -> None:
    """
    Export scalar flow values with optional bootstrap CIs.
    Rows: (metric, value, ci_lo, ci_hi)
    """
    flow = agg.get("flow", {})
    keys = ["StartPolarity", "EndPolarity", "NetShift", "Slope", "Volatility"]
    boot_scalars = (boot or {}).get("flow", {}).get("scalars", {}) or {}
    csv_path = Path(_handle_windows_path_length_limit(Path(csv_path)))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "ci_lo", "ci_hi"])
        for k in keys:
            v = float(flow.get(k, 0.0))
            b = boot_scalars.get(k)
            w.writerow([k, v, (b or {}).get("ci_lo", ""), (b or {}).get("ci_hi", "")])


def export_start_end_to_csv(agg: Dict[str, Any], csv_path: str, boot: Optional[Dict[str, Any]] = None) -> None:
    """
    Export start_probs and end_probs with optional CI.
    Rows: (type, class, value, ci_lo, ci_hi) where type in {"start", "end"}.
    """
    flow = agg.get("flow", {})
    sp = flow.get("start_probs", {}) or {}
    ep = flow.get("end_probs",   {}) or {}

    boot_sp = (boot or {}).get("flow", {}).get("start_probs", {}) or {}
    boot_ep = (boot or {}).get("flow", {}).get("end_probs",   {}) or {}

    csv_path = Path(_handle_windows_path_length_limit(Path(csv_path)))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "class", "value", "ci_lo", "ci_hi"])
        for c in CLASSES:
            b = boot_sp.get(c)
            w.writerow(["start", c, float(sp.get(c, 0.0)), (b or {}).get("ci_lo", ""), (b or {}).get("ci_hi", "")])
        for c in CLASSES:
            b = boot_ep.get(c)
            w.writerow(["end", c, float(ep.get(c, 0.0)), (b or {}).get("ci_lo", ""), (b or {}).get("ci_hi", "")])


def export_position_to_csv(agg: Dict[str, Any], csv_path: str) -> None:
    """
    Export positional mean and density per class from agg["flow"].
    Rows: ('mean', class, '', value) and ('density', class, bin, value).
    (No bootstrap CI for densities in current bootstrap output.)
    """
    flow = agg["flow"]
    means = flow["mean_position"]
    dens = flow["pos_density"]
    csv_path = Path(_handle_windows_path_length_limit(Path(csv_path)))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "class", "bin", "value"])
        for c in CLASSES:
            w.writerow(["mean", c, "", float(means[c])])
        for c in CLASSES:
            for b, val in enumerate(dens[c]):
                w.writerow(["density", c, b, float(val)])


# --------------------------- Plotters ---------------------------
def plot_stm_heatmap(stm: np.ndarray, out_path: Optional[str] = None) -> None:
    """Plot NxN heatmap of the STM (P(next|current)). If out_path is provided, saves PNG."""
    import matplotlib.pyplot as plt
    stm = np.asarray(stm, dtype=float)
    n = stm.shape[0]

    classes_to_abbr = {
        "Cooperative": "Coop",
        "Defensive": "Defe",
        "Hedge": "Hed",
        "Social": "Soc",
        "NonArg": "Oth",}
    abbr_classes = [classes_to_abbr.get(c, c) for c in CLASSES]

    fig, ax = plt.subplots(figsize=(3.5, 3))
    im = ax.imshow(stm, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.tick_params(axis='both', labelsize=14)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(abbr_classes[:n], ha="center")
    ax.set_yticklabels(abbr_classes[:n])
   
    # annotate cells
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{stm[i, j]:.2f}", ha="center", va="center", color="black", fontsize=14)

    #fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def plot_positional_density(pos_density: Dict[str, List[float]], out_path: Optional[str] = None) -> None:
    """Plot per-class positional densities as line plots over relative-position bins."""
    import matplotlib.pyplot as plt

    # detect bins from the first class found
    bins = 0
    for c in CLASSES:
        v = pos_density.get(c)
        if isinstance(v, list):
            bins = len(v)
            break

    PALETTE = {
    "Cooperative": "#2ca02c",  # green
    "Defensive":   "#d62728",  # red
    "Hedge":       "#1f77b4", # blue
    "Social":      "#ff7f0e",  # orange
    "NonArg":      "#9467bd",  # purple
    }

    x = np.linspace(0.5 / bins, 1 - 0.5 / bins, bins) if bins else []

    fig, ax = plt.subplots(figsize=(10, 7))
    for c in CLASSES:
        y = pos_density.get(c, [0.0] * bins)
        ax.plot(x, y, label=c, color=PALETTE.get(c, "#333333"), linewidth=8)

    
    ax.tick_params(axis='both', labelsize=42)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)         # fixed y-axis max
    ax.set_yticks([0.5, 1.0])
    ax.set_xticks([0.25, 0.5, 0.75])
    

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()

def plot_start_end_probs(flow: Dict[str, Any], out_path: Optional[str] = None, normalize: bool = True) -> None:
    """
    Two horizontal bins (Start, End) with class proportions stacked and colored by CLASS.
    flow must contain "start_probs" and "end_probs" dicts keyed by CLASSES.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # tab10-style colors used earlier
    COLORS = {
    "Cooperative": "#2ca02c",  # green
    "Defensive":   "#d62728",  # red
    "Hedge":       "#1f77b4", # blue
    "Social":      "#ff7f0e",  # orange
    "NonArg":      "#9467bd",  # purple
    }

    start_probs = flow.get("start_probs", {})
    end_probs   = flow.get("end_probs", {})

    start = [start_probs.get(c, 0.0) for c in CLASSES]
    end   = [end_probs.get(c, 0.0)   for c in CLASSES]

    # normalize each bar to sum to 1 (optional)
    if normalize:
        s = sum(start) or 1.0
        e = sum(end)   or 1.0
        start = [v / s for v in start]
        end   = [v / e for v in end]


    y = np.array([0, 1])
    height = 0.55
     

    fig, ax = plt.subplots(figsize=(10, 5))
    left = np.zeros_like(y, dtype=float)

    # stack segments across classes
    for c, s_val, e_val in zip(CLASSES, start, end):
        seg = np.array([s_val, e_val])
        ax.barh(y, seg, left=left, height=height,
                color=COLORS.get(c, "#999999"), edgecolor="white", linewidth=0.8,
                label=c)
        left += seg

    # labels & ticks
    ax.set_ylim(-0.5, 1.5)                 # equal space above/below the two bars
    ax.set_yticks([0, 1], ["Start", "End"])
    ax.invert_yaxis()                       # Start on top (optional)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])


    ax.tick_params(axis='both', labelsize=42)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()



def plot_TSP_class_bar(shares: Dict[str, float], title: str = "Shares", out_path: Optional[str] = None) -> None:
    """Bar plot for a single shares dict {class: value}."""
    import matplotlib.pyplot as plt
    classes = list(CLASSES)
    classes_to_abbr = {
        "Cooperative": "Coop",
        "Defensive": "Defe",
        "Hedge": "Hed",
        "Social": "Soc",
        "NonArg": "Oth",}
    abbr_classes = [classes_to_abbr.get(c, c) for c in CLASSES]
    vals = [shares.get(c, 0.0) for c in classes]

    PALETTE = {
    "Cooperative": "#2ca02c",  # green
    "Defensive":   "#d62728",  # red
    "Hedge":       "#1f77b4", # blue
    "Social":      "#ff7f0e",  # orange
    "NonArg":      "#9467bd",  # purple

    }
    colors = [PALETTE.get(c, "#333333") for c in classes]


    fig, ax = plt.subplots(figsize=(10, 7))
    # make fontsize
    ax.tick_params(axis='both', labelsize=42)
    bars = ax.bar(range(len(classes)), vals, color=colors, edgecolor="black", linewidth=0.5)
    # annotate bars
    
    for i, (b, v) in enumerate(zip(bars, vals)):
        if v > 0.01:
            ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=30)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(abbr_classes, ha="center")
    ax.set_yticks([0.5, 1.0])
    ax.set_ylim(0, 1)
    ax.set_title(title)

    # optional legend
    #ax.legend(bars, classes, frameon=False, ncol=len(classes))

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


