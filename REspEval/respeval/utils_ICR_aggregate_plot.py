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


def aggregate_and_plot_icr(path, gen_type, outdir):
    # -------- TSP and flow aggregation and plots -------
    _outdir = Path(outdir)/ 'ICR'
    _outdir.mkdir(parents=True, exist_ok=True)

    # Aggregate (macro over saved)
    agg_ICR_GPT, dicts_GPT  = aggregate_ICR_from_saved(path, gen_type, approach_name='RAG+GPT')

    agg = {'factuality_score': agg_ICR_GPT['RAG+GPT'].get('ICR', None),
           'supported_p_avg': agg_ICR_GPT['RAG+GPT'].get('supported_p', None),
           'contradicted_p_avg': agg_ICR_GPT['RAG+GPT'].get('contradicted_p', None),
           'unsupported_p_avg': agg_ICR_GPT['RAG+GPT'].get('unsupported_p', None),
           'n_decisions_avg': agg_ICR_GPT['RAG+GPT'].get('n_decisions', None),}
    print(agg)
    _outfile = _outdir / f'@{gen_type}_ICR_agg.json'
    _outfile = Path(_handle_windows_path_length_limit(_outfile))
    save_json(agg, _outfile)

    # save dicts as csv
    keys = ['sample_ix', 'ICR', 'supported_p', 'contradicted_p', 'unsupported_p', 'n_decisions']
    _outfile_all = _outdir / f'@{gen_type}_ICR_all.csv'
    _outfile_all = Path(_handle_windows_path_length_limit(_outfile_all))
    with open(_outfile_all, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(dicts_GPT)

    return agg_ICR_GPT

def aggregate_ICR_from_saved(path: str, gen_type:str, sample_subset: Optional[List[Path]] = None, approach_name:str='') -> Dict[str, object]:

    dicts = []
    base = Path(path)
    samples = sample_subset or [
        p for p in base.iterdir() if p.is_dir() and not p.name.startswith("agg_")
    ]

    for p in samples:
            subfolder = Path(p) / gen_type / 'ICR'
            file = subfolder / f'@{gen_type}_scores_{approach_name}.json'
            file = Path(_handle_windows_path_length_limit(file))
            if file.exists():
                    obj = load_json(file)
                    gen = obj.get("gen", {})
                    score = gen.get("score", {})
                    supported_p = gen.get("supported_p", {})
                    contradicted_p = gen.get("contradicted_p", {})
                    unsupported_p = gen.get("unsupported_p", {})
                    dicts.append({'sample_ix': p.name,
                         'ICR':score,
                         'supported_p': supported_p,
                         'contradicted_p': contradicted_p,
                         'unsupported_p': unsupported_p,
                            'n_decisions': gen.get("n_decisions", 0)
                                  })
    KEYS = ['ICR', 'supported_p', 'contradicted_p', 'unsupported_p', 'n_decisions']

    result = {
        approach_name:  macro_mean(dicts, KEYS) if dicts else {}
    }
    
    return result, dicts

def macro_mean(dicts: List[Dict[str, float]], keys: List[str]) -> Dict[str, float]:
    """Average keys across a list of dicts (macro). Missing keys are skipped per-key."""
    out = {}
    for k in keys:
        vals = [d.get(k) for d in dicts if d.get(k) is not None]
        if vals:
            out[k] = float(np.mean(vals))
    return out