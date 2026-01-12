from pathlib import Path
import time
import json
from .evaluate_respeval import evaluate_respeval
from .evaluate_basics import evaluate_basics
from .evaluate_politeness import evaluate_politeness
from REspEval.respeval.utils_TSP_flow_aggregate_plot import _handle_windows_path_length_limit

def evaluate_author_response(df, 
                             eval_types={'basic':None, 'politeness':None, 'respeval':['conv_spec_direct']},
                         model_path='', 
                         eval_gold=True,
                         eval_gold_model_name = '',
                         eval_pred=False, 
                         redo_atomic_facts=False, 
                         redo_knowledge_source=False):
   
    separate_save_df_cols_to_exclude = ['review_text','true','pred','system_prompt','user_input', 'user_input_wAIx','orig_output']
    all_report = {}
    
    
    for eval_type in eval_types.keys():
        print(f"Evaluating {eval_type}...")
        if eval_type == 'basic': #basic similarity-based metrics
            start_time = time.time()
            df_basic, report = evaluate_basics(df)
            end_time = time.time()
            eval_time = int(end_time - start_time)
            report['eval_totalTime'] = eval_time
            all_report = collect_all_report_and_save_df_report(df_basic, report, all_report, eval_time, model_path, separate_save_df_cols_to_exclude, eval_type)
                    
        elif eval_type == 'politeness': 
            start_time = time.time()
            #remove 'basic_' columns from df if exists
            cols_to_remove = [col for col in df.columns if col.startswith('basic_')]
            df = df.drop(columns=cols_to_remove, errors='ignore')
            df_politeness, report, sent_results = evaluate_politeness(df, eval_gold=eval_gold)
            end_time = time.time()
            eval_time = int(end_time - start_time)
            report['eval_totalTime'] = eval_time
            all_report = collect_all_report_and_save_df_report(df_politeness, report, all_report, eval_time, model_path, separate_save_df_cols_to_exclude, eval_type)
        
        elif eval_type == 'respeval': #respeval evaluation
            start_time = time.time()
            #remove 'basic_' columns from df if exists
            cols_to_remove = [col for col in df.columns if col.startswith('polite_')]
            df = df.drop(columns=cols_to_remove, errors='ignore')
            df_respeval, report = evaluate_respeval(df,
                                            gen_model_path=model_path,  # path to the generation model results to be evaluated
                                            respeval_model_name="gpt-5",  # LLM used for analysis and scoring
                                            respeval_model_key_path='.keys/azure_key.txt',
                                            eval_gold=eval_gold,
                                            eval_gold_model=eval_gold_model_name,
                                            eval_pred=eval_pred,
                                            redo_eval=True,
                                            eval_types=eval_types[eval_type]) # the list of respeval eval types to be performed
            end_time = time.time()
            eval_time = int(end_time - start_time)
            report['eval_totalTime'] = eval_time
            all_report = collect_all_report_and_save_df_report(df_respeval, report, all_report, eval_time, model_path, separate_save_df_cols_to_exclude, eval_type, save_df_to_csv=False)
        
        
def collect_all_report_and_save_df_report(df, report, all_report, eval_time, model_path, separate_save_df_cols_to_exclude=[], eval_type='', save_df_to_csv=True):
        """
        Collect all reports and save the DataFrame with the report.
        Args:
            df: DataFrame with the evaluation results
            report: dictionary with the evaluation report
            all_report: dictionary to collect all reports
            model_path: path to save the report
            separate_save_df_cols_to_exclude: columns to exclude from the saved DataFrame
        """
        all_report.update(report)
        report_json_file = model_path / f'eval_{eval_type}_report.json' # save the report to a json file
        with open(report_json_file, 'w') as f:
           json.dump(report, f, indent=4)
        if save_df_to_csv:
            basic_df_file = model_path / f'eval_{eval_type}_df.csv' # save the df with scores for individual samples
            save_df = df.drop(columns=separate_save_df_cols_to_exclude, errors='ignore')
            save_df.to_csv(basic_df_file, index=False)
        if eval_time!=0:
            time_json_file = model_path / f'eval_{eval_type}_totalTime.json' # save the time of evaluation
            with open(time_json_file, 'w') as f:
                json.dump({'eval_time': eval_time}, f, indent=4)
        return all_report







