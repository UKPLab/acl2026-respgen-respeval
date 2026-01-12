from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple, Any
from pathlib import Path
import numpy as np
import csv
import matplotlib.pyplot as plt

from REspEval.respeval.TSP import CLASSES
from REspEval.respeval.flow import _row_stochastic
from REspEval.respeval.utils_TSP_flow_aggregate_plot import save_json, load_json, macro_mean, _handle_windows_path_length_limit

def aggregate_and_plot_gfp(path, gen_type, outdir):
    # -------- TSP and flow aggregation and plots -------
    _outdir = Path(outdir)/ 'GFP'
    _outdir.mkdir(parents=True, exist_ok=True)

    # Aggregate (macro over saved)
    agg_GFP_NLI  = aggregate_GFP_from_saved(path, gen_type, approach_name='RAG+NLI')
    agg_GFP_GPT  = aggregate_GFP_from_saved(path, gen_type, approach_name='RAG+GPT')
    agg = {**agg_GFP_NLI, **agg_GFP_GPT}
    print(agg)
    _outfile = _outdir / f'@{gen_type}_GFP_agg.json'
    _outfile = Path(_handle_windows_path_length_limit(_outfile))
    save_json(agg, _outfile)

    return agg

def aggregate_GFP_from_saved(path: str, gen_type:str, sample_subset: Optional[List[Path]] = None, approach_name:str='') -> Dict[str, object]:
    
    dicts = []
    base = Path(path)
    samples = sample_subset or [
        p for p in base.iterdir() if p.is_dir() and not p.name.startswith("agg_")
    ]

    for p in samples:
            subfolder = Path(p) / gen_type / 'GFP'
            file = subfolder / f'@{gen_type}_scores_{approach_name}_final.json'
            file = Path(_handle_windows_path_length_limit(file))
            if file.exists():
                    obj = load_json(file)
                    scores = obj.get("scores", {})
                    if isinstance(scores, dict):
                        dicts.append(scores)
    KEYS = ['EditsP', 'IP', 'PP']        

    result = {
        approach_name:  macro_mean(dicts, KEYS) if dicts else {}
    }
    
    return result

def macro_mean(dicts: List[Dict[str, float]], keys: List[str]) -> Dict[str, float]:
    """Average keys across a list of dicts (macro). Missing keys are skipped per-key."""
    out = {}
    for k in keys:
        vals = [d.get(k) for d in dicts if d.get(k) is not None]
        if vals:
            out[k] = float(np.mean(vals))
    return out