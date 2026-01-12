from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

import sys
# add rootdir/RespEval to sys.path
sys.path.append(str(Path(__file__).resolve().parents[4] / "REspEval"))

from REspEval.respeval.resp_evaluater import RespEvaluator
from REspEval.respeval.utils_TSP_flow_aggregate_plot import aggregate_and_plot_tsp_flow
from REspEval.respeval.utils_GFP_aggregate_plot import aggregate_and_plot_gfp
from REspEval.respeval.utils_ICR_aggregate_plot import aggregate_and_plot_icr
from REspEval.respeval.utils_conv_spec_direct_aggregate_plot import aggregate_and_plot_conv_spec_direct
from REspEval.respeval.utils_len_control_aggregate_plot import aggregate_and_plot_len_control
from REspEval.respeval.utils_plan_aggregate_plot import aggregate_and_plot_plan
from REspEval.respeval.utils_TSP_flow_aggregate_plot import _handle_windows_path_length_limit
from REspEval.respeval.utils_factuality_aggregate_plot import aggregate_and_plot_factuality

import os, stat, shutil
import logging

logging.basicConfig(
    level=logging.INFO,                      # show INFO and above
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,                       # send to stdout (default is stderr)
    force=True                               # override existing handlers (Py>=3.8)
)

def remove_path(p: Path):
    if not p.exists():
        return

    if p.is_file() or p.is_symlink():
        try:
            p.unlink()
        except PermissionError:
            # make writable, then retry
            os.chmod(p, stat.S_IWRITE)
            p.unlink()
    else:
        # it's a directory
        def onerror(func, path, exc_info):
            # clear read-only and retry
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(p, onerror=onerror)

def evaluate_respeval(df,
                        gen_model_path='', # path to the generation model results to be evaluated
                        respeval_model_name="gpt-5", #LLM used for analysis and scoring
                        respeval_model_key_path='.keys/azure_key.txt',
                        eval_gold=False,
                        eval_gold_model='',
                        eval_pred=True,
                        redo_eval=False,
                        eval_types=['conv_spec_direct']):# a list of eval types to be performed
    
    # evaluate and analyze responses using GPT-5, extract items, find linked responses
    logging.info(f"+++++++ Starting evaluation and analysis of responses using Respeval...")
    eval_results_folder = _evaluate_analyze_response(df,
                                   eval_gold=eval_gold,
                                   eval_gold_model=eval_gold_model, #if eval_gold is False, this is used to get the gold response analysis results
                                   eval_pred=eval_pred,
                                   gen_model_path=gen_model_path,
                                   respeval_model_name=respeval_model_name,
                                   respeval_model_key_path=respeval_model_key_path,
                                   redo_eval=redo_eval)
    logging.info(f"+++++++ Done: {eval_results_folder}.")
    # get scores from the evaluation and analysis results
    logging.info(f"+++++++ Getting scores from the evaluation and analysis results...")
    _get_scores(df, eval_results_folder, respeval_model_name=respeval_model_name,
                eval_gold=eval_gold, eval_pred=eval_pred, eval_types=eval_types, eval_gold_model=eval_gold_model)
    logging.info(f"+++++++ Done.")

    df, report = add_respeval_columns_to_df(df, eval_results_folder, respeval_model_name, eval_gold=eval_gold)

    return df, report

def _get_scores(df, eval_results_folder,
                        respeval_model_name="gpt-5",
                        eval_gold=True,
                        eval_pred=True,
                        eval_types=['conv_spec_direct'],
                        eval_gold_model=''):
    
    resp_scorer = RespEvaluator(key_path='.keys/azure_key.txt', respeval_model_name=respeval_model_name)
    for i, row in tqdm(df.iterrows()):
        ix = row['chunk_ix']
        ix_folder = Path(eval_results_folder) / str(ix)
        curr_ix = {'id':i, 'ix':row['chunk_ix'], 'task':'respeval: get_scores'}
        curr_file = Path(eval_results_folder) / f"current_info.json"
        with open(curr_file, 'w') as f:
            json.dump(curr_ix, f, indent=4)
        if eval_gold: # gold responses are available, and it is not evaluated
            ix_gold_folder = ix_folder / 'gold'
            logging.info(f"############## {i+1}th sample: {ix}   ====gold")
            resp_scorer.get_scores(ix_gold_folder, eval_types, eval_gold_model=eval_gold_model)
            
        if eval_pred:
            logging.info(f"############## {i+1}th sample: {ix}   ====pred")
            ix_pred_folder = ix_folder / 'pred'
            resp_scorer.get_scores(ix_pred_folder, eval_types, eval_gold_model=eval_gold_model)
    # aggregate and plot the scores
    if eval_gold:
        gen_type = 'gold'
        outdir_gold = Path(eval_results_folder) / 'agg_gold'
        outdir_gold.mkdir(parents=True, exist_ok=True)
        aggregate_and_plot_tsp_flow(eval_results_folder, gen_type, outdir_gold)
        #aggregate_and_plot_gfp(eval_results_folder, gen_type, outdir_gold)
        
        aggregate_and_plot_factuality(eval_results_folder, gen_type, outdir_gold)
        aggregate_and_plot_conv_spec_direct(eval_results_folder, gen_type, outdir_gold)
        if 'ICR' in eval_types:
            aggregate_and_plot_icr(eval_results_folder, gen_type, outdir_gold)
        if 'len_control' in eval_types:
            aggregate_and_plot_len_control(eval_results_folder, gen_type, outdir_gold)
        if 'plan' in eval_types:
            aggregate_and_plot_plan(eval_results_folder, gen_type, outdir_gold)

    if eval_pred:
        gen_type = 'pred'
        outdir_pred = Path(eval_results_folder) / 'agg_pred'
        outdir_pred.mkdir(parents=True, exist_ok=True)
        aggregate_and_plot_tsp_flow(eval_results_folder, gen_type, outdir_pred)
        #aggregate_and_plot_gfp(eval_results_folder, gen_type, outdir_pred)
        
        aggregate_and_plot_factuality(eval_results_folder, gen_type, outdir_pred)
        aggregate_and_plot_conv_spec_direct(eval_results_folder, gen_type, outdir_pred)
        if 'ICR' in eval_types:
            aggregate_and_plot_icr(eval_results_folder, gen_type, outdir_pred)
        if 'len_control' in eval_types:
            aggregate_and_plot_len_control(eval_results_folder, gen_type, outdir_pred)
        if 'plan' in eval_types:
            aggregate_and_plot_plan(eval_results_folder, gen_type, outdir_pred)

def _evaluate_analyze_response(df,
                        eval_gold=False,
                        eval_gold_model='',
                        eval_pred=True,
                        gen_model_path='',
                        respeval_model_name="gpt-5",
                        respeval_model_key_path='.keys/azure_key.txt',
                        redo_eval=False):
    # Get the name of the model to be evaluated
    model_name = gen_model_path.name
    # Create a subfolder to save the evaluation results of the generation model
    eval_results_model_dir = '.cache/respeval/eval_results'+ f'/{model_name}'
    Path(eval_results_model_dir).mkdir(parents=True, exist_ok=True)

    for i, row in tqdm(df.iterrows()):
        #create a subfolder for each sample with unique chunk_ix
        ix = row['chunk_ix']
        ix_folder = Path(eval_results_model_dir) / str(ix)
        ix_folder = Path(_handle_windows_path_length_limit(ix_folder))
        ix_folder.mkdir(parents=True, exist_ok=True)
        # save the current info to a json file for tracking
        current_info = {'i':i, 'ix': ix, 'task':'respeval: evaluate_analyze_response'}
        file = Path(eval_results_model_dir) / f"current_info.json"
        with open(file, 'w') as f:
            json.dump(current_info, f, indent=4)
        

        ##### get review items from the review comments
        resp_scorer = RespEvaluator(key_path=respeval_model_key_path, respeval_model_name=respeval_model_name)
        if eval_gold: # gold responses are available, and it is not evaluated
            review_comment = row['review_text'].strip()
            response = row['true'].strip()
            ix_gold_folder = ix_folder / 'gold'
            ix_gold_folder = Path(_handle_windows_path_length_limit(ix_gold_folder))
            ix_gold_folder.mkdir(parents=True, exist_ok=True)
            done_folder = ix_gold_folder / f"{respeval_model_name}_done"
            done_folder = Path(_handle_windows_path_length_limit(done_folder))

            
            if done_folder.exists() and not redo_eval:
                print(f"========evaluate_analyze_response (eval_gold): Already evaluated gold sample {ix}, skipping.")
            else:
                if done_folder.exists():
                    print(f"========evaluate_analyze_response (eval_gold): Redoing evaluation for gold sample {ix}.")
                    remove_path(done_folder)
                else:
                    print(f"========evaluate_analyze_response: Evaluating gold sample {ix}")

                resp_scorer.evaluate_analyze_response(
                    review_comment,
                    response,
                    eval_mode='item-link-label-score',
                    review_items=None,
                    save_path=ix_gold_folder,
                    cost_estimate=True
                )
                done_folder.mkdir(parents=True, exist_ok=True)
            eval_gold_model = Path(eval_results_model_dir).name
        
        if eval_pred:
            review_comment = row['review_text'].strip()
            response = row['pred'].strip()
            ix_pred_folder = ix_folder / 'pred'
            ix_pred_folder = Path(_handle_windows_path_length_limit(ix_pred_folder))
            ix_pred_folder.mkdir(parents=True, exist_ok=True)
            # get the gold response analysis results 
            gold_output_file = Path('.cache/respeval/eval_results') / f'{eval_gold_model}' / f'{ix}' / 'gold' / f'{respeval_model_name}_output_dict.json'
            with open(gold_output_file, 'r') as f:
                gold_output = json.load(f)
            # clean the gold output, keep only the items, not the response and scores
            for k, v in gold_output.items():
                for _v in v:
                    _v['response'] = []
                    _v['response_spec_score'] = 0.0
                    _v['response_conv_score'] = 0.0
            gold_output['other_responses'] = []

            done_folder = ix_pred_folder / f"{respeval_model_name}_done"
            done_folder = Path(_handle_windows_path_length_limit(done_folder))
            if done_folder.exists() and not redo_eval:
                print(f"========evaluate_analyze_response (eval_pred): Already evaluated pred sample {ix}, skipping.")
            else:
                if done_folder.exists():
                    print(f"========evaluate_analyze_response (eval_pred): Redoing evaluation for pred sample {ix}.")
                    remove_path(done_folder)
                else:
                    print(f"========evaluate_analyze_response: Evaluating pred sample {ix}")
                    # use the identified items from the gold output for the evaluation of the predicted response, make sure that the items are the same
                    resp_scorer.evaluate_analyze_response(review_comment, 
                                                  response, 
                                                  eval_mode='link-label-score', 
                                                  review_items=gold_output, 
                                                  save_path=ix_pred_folder, 
                                                  cost_estimate=True)
                    # create a done folder
                    done_folder.mkdir(parents=True, exist_ok=True)

    return eval_results_model_dir



def add_respeval_columns_to_df(df, eval_results_folder, respeval_model_name, eval_gold=False):
    """
    collect evaluation results from the eval_results_folder and add them to the df
    """     
    _df = df.copy()
    eval_results_folder = Path(eval_results_folder)

    def _read_json(p: Path):
        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning(f"Missing file: {p}")
            return None
        except json.JSONDecodeError as e:
            logging.warning(f"Bad JSON in {p}: {e}")
            return None
        except Exception as e:
            logging.warning(f"Error reading {p}: {e}")
            return None

    eval_gens = ['gold', 'pred'] if eval_gold else ['pred']
    col_prefix = "respeval_"

    for i, row in tqdm(_df.iterrows(), total=len(_df), desc="Adding RespEval columns"):
        ix = row.get('chunk_ix')
        if ix is None:
            logging.warning(f"Row index {i} has no 'chunk_ix'; skipping.")
            continue

        ix_folder = eval_results_folder / str(ix)
        ix_folder = Path(_handle_windows_path_length_limit(ix_folder))

        for eval_gen in eval_gens:
            meta_fp = ix_folder / eval_gen / 'meta' / f"{respeval_model_name}_output_scores_meta.json"
            tsp_fp  = ix_folder / eval_gen / 'TSP_flow' / f"{respeval_model_name}_output_scores_TSP.json"
            tsp_fp = Path(_handle_windows_path_length_limit(tsp_fp))
            meta_fp = Path(_handle_windows_path_length_limit(meta_fp))

            scores_meta = _read_json(meta_fp) or {}
            scores_tsp  = _read_json(tsp_fp) or {}

            # Meta scores
            for k, v in scores_meta.items():
                if isinstance(v, list):
                    continue  # skip list fields
                col_name = f"{col_prefix}meta_{k}_{eval_gen}"
                if isinstance(v, (int, float, np.floating)):
                    _df.loc[i, col_name] = round(float(v), 4)
                else:
                    _df.loc[i, col_name] = str(v)

                if k == 'ph_count':
                    _df.loc[i, f"{col_prefix}meta_ph_{eval_gen}"] = (
                        1 if isinstance(v, (int, float, np.floating)) and v > 0 else 0
                    )

            # TSP scores
            tsp_block = scores_tsp.get('TSP_word_weighted', {})
            if isinstance(tsp_block, dict):
                for k, v in tsp_block.items():
                    col_name = f"{col_prefix}tsp_{k}_{eval_gen}"
                    if isinstance(v, (int, float, np.floating)):
                        _df.loc[i, col_name] = round(float(v), 4)
                    else:
                        _df.loc[i, col_name] = str(v)
            else:
                logging.warning(f"'TSP_word_weighted' missing/not a dict in {tsp_fp}")

    # save the df to a csv file
    #_df.to_csv(Path(eval_results_folder) / 'df_respeval.csv', index=False)

    # json file with the report
    gen_model_name = Path(eval_results_folder).name
    
    print(gen_model_name)
    report = get_respeval_report_from_df(_df, gen_model_name, eval_results_folder, eval_gold)
    
    return _df, report

def get_respeval_report_from_df(df, gen_model_name, eval_results_folder, eval_gold):
    respeval_meta_cols = [col for col in df.columns if col.startswith('respeval_meta_')]
    report = {}
    report['gen_model_name'] = gen_model_name
    report['sample_count'] = len(df)

    for col in respeval_meta_cols:
        # if values are not none
        if df[col].isnull().all():
            print(f"EAR_Column {col} has no values, skipping.")
            report[col] = None
        elif col in ['respeval_meta_ph_gold', 'respeval_meta_ph_pred']:
            # get the percentage of responses with placeholders
            count_with_ph = df[df[col] == 1].shape[0]
            perc_with_ph = round(count_with_ph / len(df) * 100, 2)
            report[col] = perc_with_ph
        else:
            df_with_values = df[df[col].notnull()]
            avg_score = round(df_with_values[col].mean(), 4)
            report[col] = avg_score
    # get the aggregated TSP flow metrics
    tsp_cols = [col for col in df.columns if col.startswith('respeval_tsp_')]
    tsp_cols = [col for col in tsp_cols if ('Polarity' not in col) and ('Intensity' not in col) and ('Entropy' not in col)]  
    for col in tsp_cols:
        k = col.split('_')[2]  # get the metric name
        eval_gen = col.split('_')[-1]  # get the eval_gen name
        # get the agg file name
        agg_file = Path('.cache/respeval/eval_results') / gen_model_name / f'agg_{eval_gen}' / 'TSP_flow' / f'TSP-flow_agg.json'
        if not agg_file.exists():
            print(f"TSP agg file {agg_file} does not exist, skipping.")
            continue
        agg =  json.load(open(agg_file, 'r'))
        tsp = agg['TSP']['word_weighted']
        if k == 'NonArg':
            col = f'respeval_tsp_Other_{eval_gen}'
        if k in tsp:
            report[col] = round(tsp[k],4)
        else:
            report[col] = None
    
    # get the aggregated GFP factuality metrics
    eval_gens = ['gold', 'pred'] if eval_gold else ['pred']
    for eval_gen in eval_gens:
        agg_file = Path('.cache/respeval/eval_results') / gen_model_name / f'agg_{eval_gen}' / 'factuality' / f'@{eval_gen}_factuality_agg.json'
        if not agg_file.exists():
            continue
        agg =  json.load(open(agg_file, 'r'))
        for k, v in agg.items():
            if 'factuality_score' in k:
                continue
            col = f'respeval_GFP_{k}_{eval_gen}'
            report[col] = round(v,4)

    # get the aggregated ICR metrics
    for eval_gen in eval_gens:
        agg_file = Path('.cache/respeval/eval_results') / gen_model_name / f'agg_{eval_gen}' / 'ICR' / f'@{eval_gen}_ICR_agg.json'
        if not agg_file.exists():
            continue
        agg =  json.load(open(agg_file, 'r'))
        for k, v in agg.items():
            if 'factuality_score' in k:
                continue
            col = f'respeval_ICR_{k}_{eval_gen}'
            report[col] = round(v,4)

    # get the aggregated length control metrics
    for eval_gen in eval_gens:
        agg_file = Path('.cache/respeval/eval_results') / gen_model_name / f'agg_{eval_gen}' / 'len_control' / f'@{eval_gen}_len_control_agg.json'
        if not agg_file.exists():
            continue
        agg =  json.load(open(agg_file, 'r'))
        for k, v in agg.items():
            col = f'respeval_lenC_{k}_{eval_gen}'
            report[col] = round(v,4)

    # get the aggregated plan control metrics
    for eval_gen in eval_gens:
        agg_file = Path('.cache/respeval/eval_results') / gen_model_name / f'agg_{eval_gen}' / 'plan' / f'@{eval_gen}_plan_agg.json'
        if not agg_file.exists():
            continue
        agg =  json.load(open(agg_file, 'r'))
        data = agg.get('overall_macro', {})
        data = data.get('actual_vs_plan', {})
        metrics = ['label_precision', 'label_recall', 'label_f1', 'order_fidelity']
        for k in metrics:
            col = f'respeval_planC_{k}_{eval_gen}'
            report[col] = round(data.get(k, 0),4)

    # get the aggregated quality (conv_spec_direct) metrics
    for eval_gen in eval_gens:
        agg_file = Path('.cache/respeval/eval_results') / gen_model_name / f'agg_{eval_gen}' / 'conv_spec_direct' / f'@{eval_gen}_conv_spec_direct_agg.json'
        if not agg_file.exists():
            continue
        agg =  json.load(open(agg_file, 'r'))
        col = f'respeval_quality_targeting_p_{eval_gen}'
        report[col] = round(agg.get('directness_p', 0),4)
        metrics = [
            "specificity_p",
            "convincingness_p"
        ]
        for k in metrics:
            col = f'respeval_quality_{k}_{eval_gen}'
            report[col] = round(agg.get(k, 0),4)

    #order the report by keys, group by gold and pred (if keys contain 'gold' or 'pred')
    gold_report = {k: v for k, v in report.items() if 'gold' in k }
    pred_report = {k: v for k, v in report.items() if 'pred' in k }
    other_report = {k: v for k, v in report.items() if 'gold' not in k and 'pred' not in k }
    report = {**other_report, **gold_report, **pred_report}

    return report

