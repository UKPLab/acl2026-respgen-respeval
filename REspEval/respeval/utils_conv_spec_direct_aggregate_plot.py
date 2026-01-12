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
from REspEval.respeval.utils_json_output_process import load_json_robust

def aggregate_and_plot_conv_spec_direct(path, gen_type, outdir):
    # -------- TSP and flow aggregation and plots -------
    _outdir = Path(outdir)/ 'conv_spec_direct'
    _outdir.mkdir(parents=True, exist_ok=True)

    # Aggregate (macro over saved)
    agg, dicts  = aggregate_scores_from_saved(path, gen_type)
    print(agg)
    # normalize to [0,1]
    for k in ['directness', 'specificity', 'convincingness']:
        if k in agg:
            agg[f'{k}_p'] = round(agg[k] / 5, 4)
    _outfile = _outdir / f'@{gen_type}_conv_spec_direct_agg.json'
    _outfile = Path(_handle_windows_path_length_limit(_outfile))
    save_json(agg, _outfile)

    # save dicts as csv
    keys = ['sample_ix', 'directness', 'specificity', 'convincingness']
    _outfile_all = _outdir / f'@{gen_type}_conv_spec_direct_all.csv'
    _outfile_all = Path(_handle_windows_path_length_limit(_outfile_all))
    with open(_outfile_all, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(dicts)

    # get improvement rate if it is a refinement round
    print(Path(path).name)
    if 'refine-' in Path(path).name:
        # If it is a refinement round, calculate the improvement rate
        refine_type = Path(path).name.split('_')[-2]
        print(f"Refinement type: {refine_type}")
        before_refine_model = Path(path).name.replace(f'_{refine_type}', '')
        before_refine_model = Path(path).parent / before_refine_model
        print(f"Before refine model: {before_refine_model}")
        before_refine_score_all_file = before_refine_model / f'agg_{gen_type}' / 'conv_spec_direct' / f'@{gen_type}_conv_spec_direct_all.csv'
        before_refine_score_all_file = Path(_handle_windows_path_length_limit(before_refine_score_all_file))
        if before_refine_score_all_file.exists():
            before_refine_scores = pd.read_csv(before_refine_score_all_file)
            after_refine_scores = pd.read_csv(_outfile_all)
            # merge on sample_ix
            merged = pd.merge(before_refine_scores, after_refine_scores, on='sample_ix', suffixes=('_before', '_after'))
            # save merged file
            merged_file = _outdir / f'@{gen_type}_conv_spec_direct_before_after_refine_merged.csv'
            merged_file = Path(_handle_windows_path_length_limit(merged_file))
            merged['directness_change'] = merged['directness_after'] - merged['directness_before']
            merged['specificity_change'] = merged['specificity_after'] - merged['specificity_before']
            merged['convincingness_change'] = merged['convincingness_after'] - merged['convincingness_before']
            merged.to_csv(merged_file, index=False)
            improvements = {}
            for k in ['directness', 'specificity', 'convincingness']:
                change_col = f'{k}_change'
                if change_col in merged.columns:
                    improvement_rate = (merged[change_col] > 0).sum() / len(merged)
                    improvements[f'{k}_improvement_rate'] = round(improvement_rate*100, 2)
                    keep_rate = (merged[change_col] == 0).sum() / len(merged)
                    improvements[f'{k}_keep_rate'] = round(keep_rate*100, 2)
                    decline_rate = (merged[change_col] < 0).sum() / len(merged)
                    improvements[f'{k}_decline_rate'] = round(decline_rate*100, 2)
            # rate of any improvement
            any_improvement_rate = ( (merged['directness_change'] > 0) | (merged['specificity_change'] > 0) | (merged['convincingness_change'] > 0) ).sum() / len(merged)
            improvements['any_improvement_rate'] = round(any_improvement_rate*100, 2)
            print("Improvement rates after refinement:")
            print(improvements)
            # save improvement rates
            agg.update(improvements)
            save_json(agg, _outfile)

    return agg

def aggregate_scores_from_saved(path: str, gen_type:str, sample_subset: Optional[List[Path]] = None) -> Dict[str, object]:
    
    dicts = []
    base = Path(path)
    samples = sample_subset or [
        p for p in base.iterdir() if p.is_dir() and not p.name.startswith("agg_")
    ]


    for p in samples:
            subfolder = Path(p) / gen_type / 'conv_spec_direct'
            file = subfolder / f'@{gen_type}_conv_spec_direct.json'
            file = Path(_handle_windows_path_length_limit(file))
            if file.exists():
                    obj = load_json(file)
                    if isinstance(obj, str):
                        obj = load_json_robust(obj)
                    overall = obj.get("overall", {})
                    dicts.append({'sample_ix':p.stem,
                                  "directness": overall.get("directness", None),
                                  "specificity": overall.get("specificity", None), 
                                  "convincingness": overall.get("convincingness", None),})
    KEYS = ['directness', 'specificity', 'convincingness']        

    result = macro_mean(dicts, KEYS) if dicts else {}
    return result, dicts

def macro_mean(dicts: List[Dict[str, float]], keys: List[str]) -> Dict[str, float]:
    """Average keys across a list of dicts (macro). Missing keys are skipped per-key."""
    out = {}
    for k in keys:
        vals = [d.get(k) for d in dicts if d.get(k) is not None]
        if vals:
            out[k] = float(np.mean(vals))
    return out