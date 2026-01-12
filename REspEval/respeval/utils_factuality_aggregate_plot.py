from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple, Any
from pathlib import Path
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd

from REspEval.respeval.TSP import CLASSES
from REspEval.respeval.flow import _row_stochastic
from REspEval.respeval.utils_TSP_flow_aggregate_plot import save_json, load_json, macro_mean, _handle_windows_path_length_limit


def aggregate_and_plot_factuality(path, gen_type, outdir):
    # -------- aggregation and plots -------
    _outdir = Path(outdir)/ 'factuality'
    _outdir.mkdir(parents=True, exist_ok=True)

    # Aggregate (macro over saved)
    agg_factuality_GPT, dicts_GPT = aggregate_factuality_from_saved(path, gen_type, approach_name='RAG+GPT')
    agg = {'factuality_score': agg_factuality_GPT['RAG+GPT'].get('factuality', None),
           'supported_p_avg': agg_factuality_GPT['RAG+GPT'].get('supported_p', None),
           'contradicted_p_avg': agg_factuality_GPT['RAG+GPT'].get('contradicted_p', None),
           'unsupported_p_avg': agg_factuality_GPT['RAG+GPT'].get('unsupported_p', None),
           'n_decisions_avg': agg_factuality_GPT['RAG+GPT'].get('n_decisions', None),}
    print(agg)
    _outfile = _outdir / f'@{gen_type}_factuality_agg.json'
    _outfile = Path(_handle_windows_path_length_limit(_outfile))
    save_json(agg, _outfile)

    # save dicts as csv
    keys = ['sample_ix', 'factuality', 'supported_p', 'contradicted_p', 'unsupported_p', 'n_decisions']
    _outfile_all = _outdir / f'@{gen_type}_factuality_all.csv'
    _outfile_all = Path(_handle_windows_path_length_limit(_outfile_all))
    with open(_outfile_all, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(dicts_GPT)

    # get improvement rate if it is a refinement round
    print(Path(path).name)
    if 'refine-' in Path(path).name:
        # If it is a refinement round, calculate the improvement rate
        refine_type = Path(path).name.split('_')[-2]
        print(f"Refinement type: {refine_type}")
        before_refine_model = Path(path).name.replace(f'_{refine_type}', '')
        before_refine_model = Path(path).parent / before_refine_model
        print(f"Before refine model: {before_refine_model}")
        before_refine_score_all_file = before_refine_model / f'agg_{gen_type}' / 'factuality' / f'@{gen_type}_factuality_all.csv'
        before_refine_score_all_file = Path(_handle_windows_path_length_limit(before_refine_score_all_file))
        if before_refine_score_all_file.exists():
            before_refine_scores = pd.read_csv(before_refine_score_all_file)
            after_refine_scores = pd.read_csv(_outfile_all)
            # merge on sample_ix, suffix _before and _after for other columns
            merged = pd.merge(before_refine_scores, after_refine_scores, on='sample_ix', suffixes=('_before', '_after'))
            print(merged.head())
            # save merged file
            merged_file = _outdir / f'@{gen_type}_conv_spec_direct_before_after_refine_merged.csv'
            merged_file = Path(_handle_windows_path_length_limit(merged_file))
            merged['factuality_change'] = merged['factuality_after'] - merged['factuality_before']
            merged['supported_p_change'] = merged['supported_p_after'] - merged['supported_p_before']
            merged['contradicted_p_change'] = merged['contradicted_p_after'] - merged['contradicted_p_before']
            merged['unsupported_p_change'] = merged['unsupported_p_after'] - merged['unsupported_p_before']
            merged.to_csv(merged_file, index=False)
            improvements = {}
            for k in ['factuality', 'supported_p', 'contradicted_p', 'unsupported_p']:
                change_col = f'{k}_change'
                if change_col in merged.columns:
                    improvement_rate = (merged[change_col] > 0).sum() / len(merged)
                    improvements[f'{k}_increase_rate'] = round(improvement_rate*100, 2)
                    keep_rate = (merged[change_col] == 0).sum() / len(merged)
                    improvements[f'{k}_keep_rate'] = round(keep_rate*100, 2)
                    decline_rate = (merged[change_col] < 0).sum() / len(merged)
                    improvements[f'{k}_decline_rate'] = round(decline_rate*100, 2)

            print("Improvement rates after refinement:")
            print(improvements)
            # save improvement rates
            agg.update(improvements)
            save_json(agg, _outfile)

    return agg

def aggregate_factuality_from_saved(path: str, gen_type:str, sample_subset: Optional[List[Path]] = None, approach_name:str='') -> Dict[str, object]:
    """Aggregate macro factuality over a list of per-sample JSON files.

    Returns a dict with macro means.
    """
    
    dicts = []
    base = Path(path)
    samples = sample_subset or [
        p for p in base.iterdir() if p.is_dir() and not p.name.startswith("agg_")
    ]

    for p in samples:
            subfolder = Path(p) / gen_type / 'factuality'
            file = subfolder / f'@{gen_type}_scores_{approach_name}.json'
            file = Path(_handle_windows_path_length_limit(file))
            if file.exists():
                    obj = load_json(file)
                    gen = obj.get("user-input", {})
                    score = gen.get("score", {})
                    supported_p = gen.get("supported_p", {})
                    contradicted_p = gen.get("contradicted_p", {})
                    unsupported_p = gen.get("unsupported_p", {})
                    n_decisions = gen.get("n_decisions", 0)
                    dicts.append({'sample_ix':p.stem,
                         'factuality':score,
                         'supported_p':supported_p,
                         'contradicted_p':contradicted_p,
                         'unsupported_p':unsupported_p,
                         'n_decisions':n_decisions})
    KEYS = ['factuality', 'supported_p', 'contradicted_p', 'unsupported_p', 'n_decisions']        

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