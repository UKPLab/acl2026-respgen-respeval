from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple, Any
from pathlib import Path
from REspEval.respeval.utils_ICR_aggregate_plot import aggregate_ICR_from_saved
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd

from REspEval.respeval.TSP import CLASSES
from REspEval.respeval.flow import _row_stochastic
from REspEval.respeval.utils_TSP_flow_aggregate_plot import save_json, load_json, macro_mean, _handle_windows_path_length_limit


def aggregate_and_plot_len_control(path, gen_type, outdir):
    # -------- TSP and flow aggregation and plots -------
    _outdir = Path(outdir)/ 'len_control'
    _outdir.mkdir(parents=True, exist_ok=True)

    # Aggregate stats
    agg_stats  = aggregate_stats_from_saved(path, gen_type)
    _outfile = _outdir / f'@{gen_type}_len_control_all.csv'
    _outfile = Path(_handle_windows_path_length_limit(_outfile))
    agg_stats.to_csv(_outfile, index=False)

    # get scores: % where actual length <= instructed length
    met_count = agg_stats['length_control_met'].sum()
    total_count = len(agg_stats)
    met_rate = met_count / total_count if total_count > 0 else 0.0
    avg_instructed_len = agg_stats['instructed_length'].mean() if total_count > 0 else 0.0
    avg_actual_len = agg_stats['actual_length'].mean() if total_count > 0 else 0.0
    mean_diff = agg_stats['length_diff'].mean() if total_count > 0 else 0.0
    median_diff = agg_stats['length_diff'].median() if total_count > 0 else 0.0
    scores = {
        "met_rate": met_rate,
        "avg_instructed_length": avg_instructed_len,
        "avg_actual_length": avg_actual_len,
        "mean_length_diff": mean_diff,
        "median_length_diff": median_diff
    }
    _outfile = _outdir / f'@{gen_type}_len_control_agg.json'
    _outfile = Path(_handle_windows_path_length_limit(_outfile))
    save_json(scores, _outfile)
 
    return scores

def aggregate_stats_from_saved(path: str, gen_type:str, sample_subset: Optional[List[Path]] = None) -> pd.DataFrame:
    
    dicts = []
    base = Path(path)
    samples = sample_subset or [
        p for p in base.iterdir() if p.is_dir() and not p.name.startswith("agg_")
    ]

    for p in samples:
            subfolder = Path(p) / gen_type / 'len_control'
            file = subfolder / f'stats.json'
            file = Path(_handle_windows_path_length_limit(file))
            if file.exists():
                    obj = load_json(file)
                    d = {'sample_ix': p.name,
                         'gen_type': gen_type,
                         **obj}
                    dicts.append(d)
    result = pd.DataFrame(dicts) if dicts else pd.DataFrame()
    result['length_control_met'] = result['actual_length'] <= result['instructed_length']
    result['length_diff'] = result['instructed_length'] - result['actual_length']
    return result

def macro_mean(dicts: List[Dict[str, float]], keys: List[str]) -> Dict[str, float]:
    """Average keys across a list of dicts (macro). Missing keys are skipped per-key."""
    out = {}
    for k in keys:
        vals = [d.get(k) for d in dicts if d.get(k) is not None]
        if vals:
            out[k] = float(np.mean(vals))
    return out