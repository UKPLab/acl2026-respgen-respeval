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

from pathlib import Path
import json
from collections import defaultdict


def aggregate_and_plot_plan(path, gen_type, outdir):
    # -------- TSP and flow aggregation and plots -------
    _outdir = Path(outdir)/ 'plan'
    _outdir.mkdir(parents=True, exist_ok=True)

    # Aggregate stats
    agg_stats  = aggregate_stats_from_saved(path, gen_type)

    outfile = _outdir / f'@{gen_type}_plan_agg.json'
    outfile = Path(_handle_windows_path_length_limit(outfile))

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(agg_stats, f, indent=4)

    
    return agg_stats


def deep_add(acc, d):
    """Accumulate flat numeric dict d into acc (summing values)."""
    for k, v in d.items():
        if isinstance(v, (int, float)):
            acc[k] += float(v)

def avg_dict(sum_dict, n):
    return {k: (v / n if n else 0.0) for k, v in sum_dict.items()}

def aggregate_stats_from_saved(path: str, gen_type:str, sample_subset: Optional[List[Path]] = None) -> pd.DataFrame:

    categories = ["plan_vs_gold", "actual_vs_plan", "actual_vs_gold", "fulfillment"]
    sums = {cat: defaultdict(float) for cat in categories}
    counts = 0
    missing = []
    
    base = Path(path)
    samples = sample_subset or [
        p for p in base.iterdir() if p.is_dir() and not p.name.startswith("agg_")
    ]

    for p in samples:
            subfolder = Path(p) / gen_type / 'plan'
            plan_score_file = subfolder / "plan_scores.json"
            plan_score_file = Path(_handle_windows_path_length_limit(plan_score_file))
            if not plan_score_file.exists():
                missing.append(str(plan_score_file))
                continue

            try:
                with open(plan_score_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                missing.append(f"{plan_score_file} (read_error: {e})")
                continue

            macro = data.get("macro")
            if not isinstance(macro, dict):
                missing.append(f"{plan_score_file} (no 'macro')")
                continue

            # Sum metrics for each macro category
            for cat in categories:
                block = macro.get(cat, {})
                if isinstance(block, dict):
                    deep_add(sums[cat], block)

            counts += 1

    # Averages
    avgs = {cat: avg_dict(sums[cat], counts) for cat in categories}
    out = {
        "n_samples_aggregated": counts,
        "missing_or_skipped": missing,
        "overall_macro": avgs
    }
    return out
