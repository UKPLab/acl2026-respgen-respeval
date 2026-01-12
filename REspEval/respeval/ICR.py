# GFP.py
import json
import re
from pathlib import Path
import shutil
from typing import List, Dict, Any, Optional
from REspEval.respeval.utils_evaluate_response import response_label_classes
from REspEval.respeval.atomic_facts import AtomicFactGenerator
from REspEval.respeval.atomic_fact_scorer import AtomicFactScorer
from REspEval.respeval.utils_TSP_flow_aggregate_plot import _handle_windows_path_length_limit
import numpy as np

# -------------------------------
# Helpers: get atomic facts with LLM
# -------------------------------

def _get_atomic_facts(review_text, generation_texts_list, output_file):
        import time
        start_time = time.time()
        atomic_facts = []
        af_generator = AtomicFactGenerator(key_path='.keys/azure_key.txt',
                                                        model_name="gpt-5",
                                                        output_file=output_file)
        atomic_facts, total_cost = af_generator.run(generation_texts_list, review_text)
        end_time = time.time()
        total_time_seconds = round((end_time - start_time), 4)
        l1 =len(atomic_facts)  
        atomic_facts_flattened = [fact for af in atomic_facts for fact in af['facts']]
        l2 = len(atomic_facts_flattened)
        print(f"ICR: # {l2} atomic facts from {l1} texts extracted.")
        return atomic_facts, total_cost, total_time_seconds
    
def get_atomic_facts(review_text, generation_texts_list, redo_atomic_facts, file_suffix='', data_dir=None):
            n_text_kept = len(generation_texts_list) 
    
            af_file = Path(data_dir) / f"af{file_suffix}.json"
            out_file = Path(data_dir) / f"af{file_suffix}_out.json"
            af_file = Path(_handle_windows_path_length_limit(af_file))
            out_file = Path(_handle_windows_path_length_limit(out_file))

            if af_file.exists() and not redo_atomic_facts:
                print(f"ICR: Atomic facts for the text already exists, skipping generation.")
                
                with open(af_file, 'r') as f:
                    data = json.load(f)
                    response_texts_list = [af['text'] for af in data['atomic_facts']]
                    # check if the two lists are the same
                    generation_texts_list = [gen.strip() for gen in generation_texts_list]
                    response_texts_list = [text.strip() for text in response_texts_list]
                    print(f"ICR: Texts are the same. Using existing atomic facts.")
                    af = data['atomic_facts']
                    af_dict = data
            else:
                af, total_cost, total_time_seconds = _get_atomic_facts(review_text, generation_texts_list, out_file)
                facts_count = [len(f['facts']) for f in af]
                facts_count = sum(facts_count)
                with open(af_file, 'w') as f:
                            af_dict = {'review_comment': review_text, 
                                       'texts_list': generation_texts_list,
                                        'meta': {'n_text_kept': n_text_kept, 
                                                    'n_af': facts_count,
                                                    'total_cost': total_cost,
                                                    'total_time': total_time_seconds,},
                                        'atomic_facts': af}
                            json.dump(af_dict, f, indent=4)
            return af, af_dict

# -------------------------------
# Public API
# -------------------------------
def input_coverage_recall_analysis(output,
                                             data_dir: Path,
                                             input_dict: Dict[str, Optional[str]],
                                             gen_type='',
                                             review='',
                                             score_approach_list= [], 
                                             knowledge_source_dict = {},
                                             redo_atomic_facts=False,  
                                             redo_eval=False) -> Dict[str, Any]:
    """
    """
    #### initialize scorer
    score_dir = data_dir / 'score'
    score_dir.mkdir(parents=True, exist_ok=True)
    scorer_rag_gpt, scorer_rag_nli = None, None
    if "RAG+NLI" in score_approach_list:
        scorer_rag_nli = AtomicFactScorer(approach_name="RAG+NLI", retriever_embed="specter2", device="cuda", data_dir=str(data_dir)+'/score')
    if "RAG+GPT" in score_approach_list:
        scorer_rag_gpt = AtomicFactScorer(approach_name="RAG+GPT", retriever_embed="specter2", data_dir=str(data_dir)+'/score', openai_key='.keys/azure_key.txt', openai_model="gpt-5")
    

    ##### get atomic facts from the response
    # make sure that they are saved and reusable
    af_folder = data_dir / 'af'
    af_folder.mkdir(parents=True, exist_ok=True)
    # get atomic facts for the inputs to be evaluated
    # if there are author inputs (edits), use them; else use the v1 paper input
    edits_input = input_dict.get('edits-input', None)
    v1_input = input_dict.get('v1-input', None)
    if edits_input is not None:
        file_suffix = f"@edits-input"
        atomic_facts, af_dict = get_atomic_facts(review, edits_input, redo_atomic_facts, file_suffix=file_suffix, data_dir=af_folder)
        af_source, af_source_name = edits_input, 'edits-input'
    else:
        file_suffix = f"@v1-input"
        atomic_facts, af_dict = get_atomic_facts(review, v1_input, redo_atomic_facts, file_suffix=file_suffix, data_dir=af_folder)
        af_source, af_source_name = v1_input, 'v1-input'
    if af_dict['meta']['n_af'] == 0:
        print(f"ICR: No atomic facts extracted for the response, skipping GFP scoring.")
        return {}
    
    
    scorer_dict = {'RAG+NLI': scorer_rag_nli, 'RAG+GPT': scorer_rag_gpt}

    ##### score
    def get_ICR_scores(scorer, af_source, af_source_name, af, knowledge_source_dict):
         scores = {}
         for ks_name in list(knowledge_source_dict.keys()):
                   
                    #out = scorer.get_score(review, af_source, af, knowledge_source=ks_name)
                    out = scorer.get_score(review,  af, knowledge_source=ks_name)
                    #make af_source_name the first key
                    out = {'af_source': af_source_name, **out}
                    scores[ks_name] = out
         return scores
    
    def update_ICR_scores(scores, out_file):
         # copy backup
         backup_file = str(out_file) + '.bak'
         shutil.copyfile(out_file, backup_file)

         decisions = scores.get('gen', {}).get('decisions', [])
         score = float(np.mean([d["is_supported"] for d in decisions])) if decisions else 0.0
         supported_p = sum(d["label"] == "supported" for d in decisions) / len(decisions) if decisions else 0.0
         contradicted_p = sum(d["label"] == "contradicted" for d in decisions) / len(decisions) if decisions else 0.0
         unsupported_p = sum(d["label"] == "unsupported" for d in decisions) / len(decisions) if decisions else 0.0

         user_input_score = {
            "af_source": scores.get('gen', {}).get('af_source', ''),
            "score": score,
            "supported_p": supported_p,
            "contradicted_p": contradicted_p,
            "unsupported_p": unsupported_p,
            "n_decisions": len(decisions),
            "decisions": decisions,
            "meta": scores.get('gen', {}).get('meta', {})
        }

         scores['gen'] = user_input_score
         with open(out_file, 'w') as f:
                json.dump(scores, f, indent=4)
         return scores

    scores = {}
    for approach_name, scorer in scorer_dict.items():
        if scorer is None:
            continue
        out_file = data_dir / f"@{gen_type}_scores_{approach_name}.json"
        out_file = Path(_handle_windows_path_length_limit(out_file))
        
        if out_file.exists() and not redo_eval:
            print(f"ICR: scores already exists, skipping evaluation.")
            ICR_scores = json.load(open(out_file, 'r', encoding='utf-8'))
            if "supported_p" not in ICR_scores['gen'].keys():
                print(f"ICR: old score format detected, updating evaluation.")
                ICR_scores = update_ICR_scores(ICR_scores, out_file)
                
        else:
            ##### create knowledge sources and register them with the scorers
            for ks_name, (ks, ks_type) in knowledge_source_dict.items():
                if ks_type == 'static':
                    scorer.register_static_source(ks_name, ks)  # short list[str]
                   
                elif ks_type == 'rag':
                    scorer.register_rag_source(ks_name, ks)
                    
                else:
                    raise ValueError(f"Unknown knowledge source type: {ks_type}")

            ICR_scores = get_ICR_scores(scorer, af_source, af_source_name, atomic_facts, knowledge_source_dict)

            with open(out_file, 'w') as f:
                json.dump(ICR_scores, f, indent=4)
        scores[approach_name] = ICR_scores

    return scores        
   
    
# -------------------------------
# CLI
# -------------------------------

if __name__ == "__main__":
    print("This module is not intended to be run as a script.")
