import os
import re
import logging
import json
from pathlib import Path
import glob
import json
from pathlib import Path
from tempfile import template
from typing import Dict, List, Sequence, Any
import pandas as pd

from REspEval.respeval.openai_lm import OpenAIModel
from REspEval.respeval.utils_evaluate_response import system_prompt_dict
from REspEval.respeval.TSP import tone_stance_profile_multilabel
from REspEval.respeval.flow import class_transition_position_analysis
from REspEval.respeval.factuality import generation_factuality_analysis, process_response
from REspEval.respeval.ICR import input_coverage_recall_analysis
from REspEval.respeval.conv_spec_direct import convincingness_specificity_directness_analysis
from REspEval.respeval.plan import evaluate_plan_controllability
from REspEval.respeval.utils_json_output_process import load_json_robust
from REspEval.respeval.utils_TSP_flow_aggregate_plot import _handle_windows_path_length_limit


import logging, sys

logging.basicConfig(
    level=logging.INFO,                      # show INFO and above
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,                       # send to stdout (default is stderr)
    force=True                               # override existing handlers (Py>=3.8)
)


def get_v1_full_paper(row):
    TITLE_NTYPES = {
    "title",              # generic section/subsection title 
    "abstract",           # keep abstract heading 
    }
    ROOT_NTYPE = "article-title"   # stop climbing at this node type
    doc_name = row['doc_name']
    if doc_name.startswith('emnlp24_'):
        v1_doc_itg = Path('data_triplets') / 'emnlp24' / 'docs' / doc_name / 'revision' / 'v1.json'
    else:
        v1_doc_itg = Path('data_triplets') / 'peerj' / 'docs' / doc_name / 'revision' / 'v1.json'
    v1_doc_itg = Path(_handle_windows_path_length_limit(v1_doc_itg))
    with open(v1_doc_itg, 'r', encoding='utf-8') as f:
        v1_doc = json.load(f)

    # Fast lookups
    nodes_by_ix = {n['ix']: n for n in v1_doc['nodes']}
    parent_of = {e['tgt_ix']: e['src_ix'] for e in v1_doc['edges'] if e.get('etype') == 'parent'}

    # Paragraph nodes
    v1_doc_paras = [n for n in v1_doc['nodes'] if n.get('ntype') == 'p' and n.get('content')]

    v1_doc_with_title = []
    for node in v1_doc_paras:
        titles = []
        seen = set()
        cur_ix = node['ix']
        # climb parents up to article-title (exclusive) safely
        while cur_ix in parent_of:
            par_ix = parent_of[cur_ix]
            if par_ix in seen:
                # cycle guard; break to avoid infinite loop
                break
            seen.add(par_ix)
            par = nodes_by_ix.get(par_ix)
            if not par:
                break
            ntype = par.get('ntype')
            content = (par.get('content') or "").strip()
            # collect section/subsection/abstract titles
            if ntype in TITLE_NTYPES and content:
                titles.append(content)
            # donnot include article title, stop
            if ntype == ROOT_NTYPE:
                break
            # move up
            cur_ix = par_ix
        # Build heading from outermost → innermost
        heading = " -- ".join(reversed(titles)) if titles else ""
        if heading:
            combined = f"[Section: {heading}]\n{node['content']}"
        else:
            combined = node['content']  # fallback: paragraph only
        v1_doc_with_title.append(combined)
    
    return v1_doc_with_title


class RespEvaluator(object):

    def __init__(self,
                 respeval_model_name="gpt-5",
                 key_path="api.key",
                 cache_dir=".cache/respeval/cache_files",
                 ):

        self.respeval_model_name = respeval_model_name
        self.key_path = key_path
        self.cache_dir = cache_dir
        
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def cost_estimates(self, input_words, output_words, task, model):
        # Number of tokens are roughly 4/3 of the number of words
        total_input_tokens = input_words * 4.0 / 3
        total_output_tokens = output_words * 4.0 / 3

        if model == 'gpt-5':
            input_rate = 0.00125
            output_rate = 0.01
        elif model == 'gpt-5 mini':
            input_rate = 0.00025
            output_rate = 0.002
        elif model == "gpt-5 nano":
            input_rate = 0.00005
            output_rate = 0.0004

        input_cost = total_input_tokens * input_rate / 1000
        output_cost = total_output_tokens * output_rate / 1000
        total_cost = input_cost + output_cost

        # print the total words, tokens, and cost along with rate
        logging.info("Estimated OpenAI API cost for %s : $%.6f for %d input words and %d output words" % (task,  total_cost, input_words, output_words))
        return total_cost
    
    def evaluate_analyze_response(self,
                  review_comment,
                  response,
                  eval_mode = 'item-link-label-score', #'link-score'
                  review_items=None,
                  save_path=None,
                  cost_estimate=True,
                  redo_eval=False,
                  creat_anno_file=False
                 ):
        cache_file = Path(self.cache_dir) / f"{self.respeval_model_name}_response_analysis.db"
        lm = OpenAIModel(self.respeval_model_name, key_path=self.key_path, cache_file=cache_file)

        max_output_length= 8000

        if eval_mode == 'link-label-score':
            assert review_items is not None, "Please provide review items for link-label-score mode."
            system_prompt = system_prompt_dict[eval_mode]
            user_input = f"The review comment is: {review_comment}\nThe response is: {response}\n\
            The review items in json: {json.dumps(review_items, indent=4)}"


        elif eval_mode == 'item-link-label-score':
            system_prompt = system_prompt_dict[eval_mode]
            user_input = f"The review comment is: {review_comment}\nThe response is: {response}"
        
        meta_output_file = os.path.join(save_path, f"{self.respeval_model_name}_output_meta_cost.json")
        output_file = os.path.join(save_path, f"{self.respeval_model_name}_output_dict.json")
        output_file = Path(self._handle_windows_path_length_limit(Path(output_file)))
        meta_output_file = Path(self._handle_windows_path_length_limit(Path(meta_output_file)))

        if os.path.exists(output_file) and not redo_eval:
                logging.info(f"   - respeval:evaluate_analyze_response: Already has output {output_file}, skipping.")
                with open(output_file, 'r') as f:
                    output = json.load(f)
        else:
                    output = lm.generate(system_prompt, user_input, max_output_length=max_output_length)
                    if cost_estimate:
                        input_words = len(review_comment.split()) + len(response.split())
                        output_words = len(output.split())
                        total_cost = self.cost_estimates(input_words, output_words, 'respeval_response analysis', self.respeval_model_name)
                    # save the raw output for debugging just in case output is not json format
                    output_file2 = str(output_file).replace('_dict.json', '_dictB.json')
                    with open(output_file2, 'w') as f:
                        json.dump(output, f, indent=4)
                    output = load_json_robust(output)

                    if save_path is not None:
                        with open(meta_output_file, 'w') as f:
                            json.dump({'model': self.respeval_model_name,
                                    "input_words": input_words,
                                    "output_words": output_words,
                                    "total_cost": total_cost if cost_estimate else None}, f, indent=4)
                        with open(output_file, 'w') as f:
                            json.dump(output, f, indent=4)

                        # save output dict with review comment and response
                        chunk_ix = save_path.parent.name
                        data = {
                            'chunk_ix': chunk_ix,
                            "review_comment": review_comment,
                            "response": response,
                            "items": output
                        }
                        file = str(output_file).replace('_dict.json', '_dict_with_input.json')
                        with open(file, 'w') as f:
                            json.dump(data, f, indent=4)


                        # generate a json file for human annotation, optional
                        if not creat_anno_file:
                            return 
                        for k, v in output.items():
                            if k!= 'other_responses':
                                for _v in v:
                                    for resp in _v['response']:
                                        resp['ANNO_response correct'] = ''
                                        resp['ANNO_response label correct'] = ''
                                    _v['ANNO_item correct'] = ''
                                    _v['ANNO_response convincing score reasonable'] = ''
                                    _v['ANNO_response specificity score reasonable'] = ''
                            else:
                                for _v in v:
                                        _v['ANNO_response correct'] = ''
                                        _v['ANNO_response label correct'] = ''
                                   
                        data = {
                            "chunk_ix": chunk_ix,
                            "review_comment": review_comment,
                            "response": response,
                            "items": output
                        }
                        anno_file = str(output_file).replace('_dict.json', '_ANNO.json')
                        with open(anno_file, 'w') as f:
                            json.dump(data, f, indent=4)

    
    def get_scores(self, data_dir, eval_types, eval_gold_model=''):
        output_file = data_dir / f"{self.respeval_model_name}_output_dict.json"
        output_file = Path(self._handle_windows_path_length_limit(output_file))
        with open(output_file, 'r') as f:
             output = json.load(f)
        ix = data_dir.parents[0].name
        # get meta scores: comprehensiveness, specificity, convincingness (LLM scores), counts of items and response length
        # as well as 'other_responses' if available
        meta_scores, TSP_and_flow_scores, GFP_scores, ICR_scores, conv_spec_scores = {}, {}, {}, {}, {}
        len_control_stats, plan_scores = {}, {}
        if 'meta' in eval_types:
            logging.info(f"     -- respeval:get_scores: Getting meta scores...{ix}")
            meta_scores = self.get_meta_scores_from_output(output, output_file)
        if 'TSP_flow' in eval_types:
             # get tone-stance profile scores and interaction flow scores for each response item
            logging.info(f"     -- respeval:get_scores: Getting TSP and flow scores...{ix}")
            TSP_and_flow_scores = self.get_tone_stance_and_flow_scores_from_output(output, output_file)
        if 'factuality' in eval_types:
            # get generation factuality precision scores for each response item
            logging.info(f"     -- respeval:get_scores: Getting factuality scores...{ix}")
            fact_scores = self.get_generation_factuality_scores(output, output_file, data_dir,
                                                                     score_approach_list=["RAG+GPT"],
                                                                     redo_atomic_facts=False,
                                                                     redo_eval=False)
        if 'ICR' in eval_types:
            # get input coverage recall scores for each response item
            logging.info(f"     -- respeval:get_scores: Getting ICR scores...{ix}")
            ICR_scores = self.get_input_coverage_recall_scores(output, output_file, data_dir,
                                                                     score_approach_list=["RAG+GPT"],
                                                                     redo_atomic_facts=False,
                                                                      redo_eval=False)
         
        if 'conv_spec_direct' in eval_types:
            # get convincingness and specificity scores for each response item
            logging.info(f"     -- respeval:get_scores: Getting convincingness, specificity, directness scores...{ix}")
            conv_spec_scores = self.get_convincingness_specificity_directness_scores(output, output_file, data_dir, redo_eval=False)

        if 'len_control' in eval_types:
            # get length control scores for each response item
            logging.info(f"     -- respeval:get_scores: Getting length control scores...{ix}")
            # TO DO: implement length control scores
            len_control_stats = self.get_length_control_stats(output, output_file, data_dir, redo_eval=False)
       
        if 'plan' in eval_types:
            # get planning scores for each response item
            logging.info(f"     -- respeval:get_scores: Getting planning scores...{ix}")
            # TO DO: implement planning scores
            plan_scores = self.get_planning_scores(output, output_file, data_dir, redo_eval=False, eval_gold_model=eval_gold_model)
        
        score_list = [meta_scores, TSP_and_flow_scores, fact_scores, ICR_scores, conv_spec_scores, plan_scores]
        
        score_list = [s for s in score_list if s]
        scores = {}
        for s in score_list:
            scores.update(s)
        return scores
    
    def get_planning_scores(self, output, output_file, data_dir, redo_eval=False, eval_gold_model=''):
        plan_scores = {}
        # get row
        row, evaluated_model_name, sample_ix, gen_type = self._get_generation_row_and_meta(output_file)
        if 'plan'  not in str(evaluated_model_name):
            logging.info(f"     -- respeval:get_planning_scores: No planning used in the generation, skipping planning scores...{sample_ix}")
            return plan_scores

        # make the subfolder for plan scores
        d = output_file.parent / 'plan'
        d = Path(self._handle_windows_path_length_limit(d))
        d.mkdir(exist_ok=True, parents=True)
        # get gold human plan, LLM plan, and generated response plan
        # gold human plan
        def _get_response_plan_and_save(json_file, outfile_name, outdir):
                json_file = Path(self._handle_windows_path_length_limit(json_file))
                with open(json_file, 'r') as f:
                    output = json.load(f)
                # clean the gold output, keep only the items, not the response and scores
                for k, v in output.items():
                    if k in ['questions', 'criticisms', 'requests']:
                        for i, _v in enumerate(v):
                            response_plan = [resp['labels'] for resp in _v['response']]
                            _v['id'] = i
                            _v['response_plan'] = response_plan
                            # remove keys not needed for plan scores
                            _v.pop('response', None)
                            _v.pop('response_spec_score', None)
                            _v.pop('response_conv_score', None)
                output.pop('other_responses', None)
                outfile = outdir / f'{outfile_name}.json'
                outfile = Path(self._handle_windows_path_length_limit(outfile))
                with open(outfile, 'w') as f:
                    json.dump(output, f, indent=4)
                return output
        gold_output_file = Path('.cache/respeval/eval_results') / f'{eval_gold_model}' / f'{sample_ix}' / 'gold' / f'gpt-5_output_dict.json'
        
        gold_plan = _get_response_plan_and_save(gold_output_file, 'gold_human_plan',  d)
        # make a deep copy of gold plan for LLM plan to modify
        llm_plan = json.loads(json.dumps(gold_plan))
        # actual plan used in generation
        actual_plan = _get_response_plan_and_save(output_file, 'generation_actual_plan', d)

        #LLM plan from LLM output
        def _get_response_plan_from_template(template: Dict[str, Any], plan_str: Dict[str, str]) -> Dict[str, Any]:
            """
            Update template['criticisms'][i]['response_plan'], etc., based on plan_text['plan'].
            Assumes plan_text['plan'] has lines like:
                --- criticisms: #1: mitigate criticism, contradict assertion
                --- questions: #1: answer question; #2: accept for future work
            """
            plan_lines = [ln.strip() for ln in plan_str.splitlines() if ln.strip().startswith("---")]
            parsed_plan = {}
            for line in plan_lines:
                m = re.match(r"---\s*(\w+):\s*(.*)", line)
                if not m:
                    continue
                category, content = m.groups()
                category = category.lower()
                item_plans = {}

                parts = [p.strip() for p in content.split(";") if p.strip()]
                for p in parts:
                    m2 = re.match(r"#(\d+):\s*(.*)", p)
                    if m2:
                        idx = int(m2.group(1)) - 1  # convert to 0-based
                        # split by commas and strip punctuation
                        labels = [
                            lab.strip().rstrip(".,;:") 
                            for lab in m2.group(2).split(",") 
                            if lab.strip()
                        ]
                        item_plans[idx] = labels
                parsed_plan[category] = item_plans

            # Update the template
            for cat in ["questions", "criticisms", "requests"]:
                if cat not in template:
                    continue
                if cat in parsed_plan:
                    for idx, labels in parsed_plan[cat].items():
                        if idx < len(template[cat]):
                            # convert to single label list
                            l_list = []
                            for l in labels:
                                 l_list.append([l])
                            template[cat][idx]["response_plan"] = l_list

            return template

        if 'self-plan' in str(evaluated_model_name):
            plan = row['plan'].strip()
            llm_plan = _get_response_plan_from_template(llm_plan, plan)
            file = d / f'llm_self_plan.json'
            file = Path(self._handle_windows_path_length_limit(file))
            with open(file, 'w') as f:
                json.dump(llm_plan, f, indent=4)
        elif 'author-plan' in str(evaluated_model_name):
             llm_plan = gold_plan
             file = d / f'llm_author_plan.json'
             file = Path(self._handle_windows_path_length_limit(file))
             with open(file, 'w') as f:
                 json.dump(llm_plan, f, indent=4)

        plan_scores = evaluate_plan_controllability(gold_plan, llm_plan, actual_plan)
        file = d / f'plan_scores.json'
        file = Path(self._handle_windows_path_length_limit(file))
        with open(file, 'w') as f:
            json.dump(plan_scores, f, indent=4)

        return plan_scores
    
    def get_length_control_stats(self, output, 
                                                    output_file,
                                                    data_dir, 
                                                    redo_eval=False):
        # make the subfolder for len_control scores
        d = output_file.parent / 'len_control'
        d.mkdir(exist_ok=True, parents=True)
        # get row
        row, evaluated_model_name, sample_ix, gen_type = self._get_generation_row_and_meta(output_file)
        # get instructed length from system prompt
        system_prompt = row['system_prompt'].strip()
        instructed_length = system_prompt.split('NO MORE than')[-1].strip().split('words')[0].strip()
        instructed_length = int(instructed_length)
        # get actual response length
        pred = row['pred'].strip()
        actual_length = len(pred.split())
        # get human gold response length
        true = row['true'].strip()
        gold_length = len(true.split())
        len_control_stats = {'user_input_length': len(row['user_input'].strip().split()),
            'instructed_length': instructed_length,
            'actual_length': actual_length,
            'gold_length': gold_length,
            
        }
        file = d / f"stats.json"
        file = Path(self._handle_windows_path_length_limit(file))
        with open(file, 'w') as f:
            json.dump(len_control_stats, f, indent=4)
        return len_control_stats

    def get_convincingness_specificity_directness_scores(self, output, 
                                                    output_file, 
                                                    data_dir, 
                                                    redo_eval=False):
         # make the subfolder for conv_spec scores
         d = output_file.parent / 'conv_spec_direct'
         d.mkdir(exist_ok=True, parents=True)
         # get conv_spec_direct scores
         row, evaluated_model_name, sample_ix, gen_type = self._get_generation_row_and_meta(output_file)

         review = row['review_text'].strip()
         col_name = 'pred' if gen_type=='pred' else 'true'
         response = row[col_name].strip()
         conv_spec_scores = convincingness_specificity_directness_analysis(output,
                                                                            d, 
                                                                            redo_eval=redo_eval, 
                                                                            review=review, 
                                                                            response=response,
                                                                            gen_type=gen_type)
         return conv_spec_scores
    def get_generation_factuality_scores(self, output, 
                                                    output_file, 
                                                    data_dir, 
                                                    score_approach_list=["RAG+NLI", "RAG+GPT"],
                                                    redo_atomic_facts=False,
                                                    redo_eval=False):
        # get the eval
        d = output_file.parent / 'factuality'
        d.mkdir(exist_ok=True, parents=True)

        # get generation factuality precision scores 
        row, evaluated_model_name, sample_ix, gen_type = self._get_generation_row_and_meta(output_file)

        # get knowledge source: user_input

        ##### extract the true response, predicted response, review comment, and user input (keep only edits text with additional facts)
        true = row['true'].strip()
        pred = row['pred'].strip()
        review = row['review_text'].strip()
        user_input = row['user_input'].strip()

        review_comment = [t.replace('- The review comment is:', '').strip() for t in user_input.split('\n\n') if t.strip().startswith('- The review comment is:')]
        review_comment = [review_comment[0].split('-- The items extracted from the review comment are:')[0].strip()] if review_comment else ''
      
        if ('noAIx_' in evaluated_model_name) and ('_v1_' not in evaluated_model_name): #no author input, no v1 input
                if row['user_input_wAIx']!='':
                     _outfile = Path(str(output_file).replace('-noAIx_', '-wAIx_').replace('_temp0','_S+SecT+P+v1_temp0'))
                     print('_outfile:', _outfile)
                     _row, _, _, _ = self._get_generation_row_and_meta(_outfile)

                     user_input = _row['user_input'].strip()
                     
                     edit_texts = self._get_edit_texts_from_user_input(user_input)
                     edits_input = []
                     edits_contexts = []
                     for t in edit_texts:
                        edit_input, edit_context = self._get_edit_input_contexts_from_text(t, has_para_context=True)
                        edits_input.append(edit_input)
                        edits_contexts.append(edit_context)
                     edits_input = list(set(edits_input)) if edits_input is not None else None
                     edits_contexts = list(set(edits_contexts)) if edits_contexts is not None else None
                     v1_input = self._get_v1_input_from_user_input(user_input)
                else:
                     edits_input, edits_contexts, v1_input = None, None, None

        elif ('noAIx_' in evaluated_model_name) and ('_v1_' in evaluated_model_name): #v1 input only, no author input
                edits_input, edits_contexts = None, None
                v1_input = self._get_v1_input_from_user_input(user_input)
        else:
                
                edit_texts = self._get_edit_texts_from_user_input(user_input)
                
                if '_S+SecT+P+v1_' in evaluated_model_name:
                     edits_input = []
                     edits_contexts = []
                     for t in edit_texts:
                        edit_input, edit_context = self._get_edit_input_contexts_from_text(t, has_para_context=True)
                        edits_input.append(edit_input)
                        edits_contexts.append(edit_context)
                     v1_input = self._get_v1_input_from_user_input(user_input)
                     
                elif '_S+SecT+P_' in evaluated_model_name:
                     edits_input = []
                     edits_contexts = []
                     v1_input = None
                     for t in edit_texts:
                        edit_input, edit_context = self._get_edit_input_contexts_from_text(t, has_para_context=True)
                        edits_input.append(edit_input)
                        edits_contexts.append(edit_context)
                elif ('_S_' in evaluated_model_name):
                     edits_input = []
                     edits_contexts = None
                     v1_input = None
                     for t in edit_texts:
                        edit_input, _ = self._get_edit_input_contexts_from_text(t, has_para_context=False)
                        edits_input.append(edit_input)
                # remove any duplicate edits_input and edits_contexts
                edits_input = list(set(edits_input)) if edits_input is not None else None
                edits_contexts = list(set(edits_contexts)) if edits_contexts is not None else None
                
        
        user_input_combined = [review_comment, edits_input, edits_contexts, v1_input]
        user_input_combined = [ui for ui in user_input_combined if ui is not None]
        #flatten the list
        user_input_combined = [item for sublist in user_input_combined for item in sublist]
        
        ks_dict = {
                'user-input': (user_input_combined, 'static')}
        
        # keep only non-empty knowledge sources
        ks_dict = {k: v for k, v in ks_dict.items() if v[0] is not None}

        factuality_scores = generation_factuality_analysis(output, d, 
                                                    gen_type=gen_type,
                                                    review=review,
                                                    score_approach_list=score_approach_list,
                                                    knowledge_source_dict=ks_dict,
                                                 redo_atomic_facts=redo_atomic_facts, 
                                                 redo_eval=redo_eval)
        
        return factuality_scores
    
    def _get_generation_row_and_meta(self, output_file):
        evaluated_model_name = output_file.parents[2].name
        sample_ix = output_file.parents[1].name
        gen_type = output_file.parents[0].name

        
        root_path = Path('results/author_response_generation/inference_llm')

        gen_df_file = root_path / evaluated_model_name / 'eval_pred.csv'
        gen_df = pd.read_csv(gen_df_file)
        # get the sample row, convert to row
        gen_row = gen_df[gen_df['chunk_ix'] == str(sample_ix)]
        assert len(gen_row) == 1, f"Expected one row for chunk_ix {sample_ix}, but got {len(gen_row)}"
        row = gen_row.iloc[0]
        return row, evaluated_model_name, sample_ix, gen_type
    
    def _get_v1_input_from_user_input(self, user_input):
        v1_input_texts = user_input.split('\n\n')
        v1_input_texts = [text.strip() for text in v1_input_texts if text.strip().startswith('- Here are the top 5 paragraphs retrieved from the original paper:')]
        assert len(v1_input_texts) == 1, f"Expected at most one v1 text block, but got {len(v1_input_texts)}"
        v1_input_texts = v1_input_texts[0].replace('- Here are the top 5 paragraphs retrieved from the original paper:', '').strip().split('-- [')
        v1_input = ['['+text.strip() for text in v1_input_texts if text.strip()]
        return v1_input
    
    def _get_edit_texts_from_user_input(self, user_input):
        if user_input.endswith('Output the response only. Do not include any other text.'):
            user_input = user_input.replace('Output the response only. Do not include any other text.', '').strip()
        edit_texts = user_input.split('\n\n')
        edit_texts = [text.strip() for text in edit_texts if text.strip().startswith('- Refer to the author input below:')]
        assert len(edit_texts) == 1, f"Expected at most one edit text block, but got {len(edit_texts)}"
        edit_texts = edit_texts[0].replace('- Refer to the author input below:', '').strip().split('-- Authors will add:')
        edit_texts = [text.strip() for text in edit_texts if text.strip() and text.strip().startswith('<')]
        
        return edit_texts
    
    def _get_edit_input_contexts_from_text(self, edit_text, has_para_context=True):
        if has_para_context:
             t0 =  edit_text.split('> in a paragraph <')[0]  + '>'
             t1 = '<' +edit_text.split('> in a paragraph <')[1]
             found_texts = re.findall(r'<(.*?)>', t0, re.S)
             edit_input = found_texts[0].strip()
             found_texts = re.findall(r'<(.*?)>', t1, re.S)
             edit_context = found_texts[0].strip()
        else:
             t0 = edit_text.split('> in Section <')[0] 
             if not t0.endswith('>.'):
                    t0 = t0 + '>'
             found_texts = re.findall(r'<(.*?)>', t0, re.S)
             edit_input = found_texts[0].strip()
             edit_context = None
        return edit_input, edit_context
    
    def get_input_coverage_recall_scores(self, output, 
                                                    output_file, 
                                                    data_dir, 
                                                    score_approach_list=["RAG+NLI", "RAG+GPT"],
                                                    redo_atomic_facts=False,
                                                    redo_eval=False):
        # make the subfolder for ICR scores
        d = output_file.parent / 'ICR'
        d.mkdir(exist_ok=True, parents=True)

        # get ICR scores
        row, evaluated_model_name, sample_ix, gen_type = self._get_generation_row_and_meta(output_file)

        # get knowledge sources
        ##### extract the true response, predicted response, review comment, and user input (keep only edits text with additional facts)
        true = row['true'].strip()
        pred = row['pred'].strip()
        review = row['review_text'].strip()
        user_input = row['user_input'].strip()
        
        if ('noAIx_' in evaluated_model_name) and ('_v1_' not in evaluated_model_name): #no author input, no v1 input
                if row['user_input_wAIx']!='':
                     _outfile = Path(str(output_file).replace('-noAIx_', '-wAIx_').replace('_temp0','_S+SecT+P+v1_temp0'))
                     print('_outfile:', _outfile)
                     _row, _, _, _ = self._get_generation_row_and_meta(_outfile)

                     user_input = _row['user_input'].strip()
                     #print('######2.user_input:  \n ', user_input)
                     
                     edit_texts = self._get_edit_texts_from_user_input(user_input)
                     edits_input = []
                     edits_contexts = []
                     for t in edit_texts:
                        edit_input, edit_context = self._get_edit_input_contexts_from_text(t, has_para_context=True)
                        edits_input.append(edit_input)
                        edits_contexts.append(edit_context)
                     edits_input = list(set(edits_input)) if edits_input is not None else None
                     edits_contexts = list(set(edits_contexts)) if edits_contexts is not None else None
                     v1_input = self._get_v1_input_from_user_input(user_input)
                else:
                     gen=None
                     edits_input, edits_contexts, v1_input = None, None, None
                     return None
        elif ('noAIx_' in evaluated_model_name) and ('_v1_' in evaluated_model_name): #v1 input only, no author input
                     edits_input, edits_contexts = None, None
                     v1_input = self._get_v1_input_from_user_input(user_input)
        else:  
                edit_texts = self._get_edit_texts_from_user_input(user_input)

                if '_S+SecT+P+v1_' in evaluated_model_name:
                     edits_input = []
                     edits_contexts = []
                     for t in edit_texts:
                        edit_input, edit_context = self._get_edit_input_contexts_from_text(t, has_para_context=True)
                        edits_input.append(edit_input)
                        edits_contexts.append(edit_context)
                     v1_input = self._get_v1_input_from_user_input(user_input)
                elif '_S+SecT+P_' in evaluated_model_name:
                     edits_input = []
                     edits_contexts = []
                     v1_input = None
                     for t in edit_texts:
                        edit_input, edit_context = self._get_edit_input_contexts_from_text(t, has_para_context=True)
                        edits_input.append(edit_input)
                        edits_contexts.append(edit_context)
                elif ('_S_' in evaluated_model_name):
                     edits_input = []
                     edits_contexts = None
                     v1_input = None
                     for t in edit_texts:
                          edit_input, _ = self._get_edit_input_contexts_from_text(t, has_para_context=False)
                          edits_input.append(edit_input)
              
                # remove any duplicate edits_input and edits_contexts
                edits_input = list(set(edits_input)) if edits_input is not None else None
                edits_contexts = list(set(edits_contexts)) if edits_contexts is not None else None
                
                        

        gen_texts_list = process_response(output)
        
        ks_dict = {
                'gen': (gen_texts_list, 'static')}
        # keep only non-empty knowledge sources
        ks_dict = {k: v for k, v in ks_dict.items() if v[0] is not None}
        input_dict = {
            'edits-input': edits_input,
            'edits-contexts': edits_contexts,
            'v1-input': v1_input,
        }
       
        icr_scores = input_coverage_recall_analysis(output, d, 
                                                    input_dict=input_dict,
                                                    gen_type=gen_type,
                                                    review=review,
                                                    score_approach_list=score_approach_list,
                                                    knowledge_source_dict=ks_dict,
                                                 redo_atomic_facts=redo_atomic_facts, 
                                                 redo_eval=redo_eval)
        
        return icr_scores
    
    
    def _handle_windows_path_length_limit(self, path: Path) -> str:
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
    
    def get_tone_stance_and_flow_scores_from_output(self, output, output_file):
        for k, v in output.items():
            if k != 'other_responses':
                # get TSP and flow scores for each item
                sent_labels = []
                word_lengths = []
                for _v in v:
                    for resp in _v['response']:
                        sent_labels.append(resp['labels'])
                        word_lengths.append(len(resp['text'].split()))
                    profiles = tone_stance_profile_multilabel(sent_labels, word_lengths, cap_words=150, sublinear=None)
                    
                    flow = class_transition_position_analysis(sent_labels)
                    _v_scores = {'TSP_sentence_weighted': profiles["sentence_weighted"],
                                'TSP_word_weighted': profiles["word_weighted"],
                                'Transition_flow': flow}
                    _v['respeval_scores'] = _v_scores
        d = output_file.parent / 'TSP_flow'
        d.mkdir(exist_ok=True, parents=True)
        file = str(output_file.name).replace('_dict.json', '_dict_with_TSP_and_flow.json')
        file = d / file
        
        # deal with potential Windows path length limitation
        file = self._handle_windows_path_length_limit(file)
        with open(file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

        # calculate overall TSP scores, including 'other_responses' if available
        all_response_sents = [resp for k, v in output.items() if k!= 'other_responses' for _v in v for resp in _v['response']]
        other_response_sents = output.get('other_responses', [])
        all_response_sents.extend(other_response_sents)
        all_sent_labels = [resp['labels'] for resp in all_response_sents]
        all_word_lengths = [len(resp['text'].split()) for resp in all_response_sents]
        overall_profiles = tone_stance_profile_multilabel(all_sent_labels, all_word_lengths, cap_words=150, sublinear=None)
        
        overall_TSP_scores = {'TSP_sentence_weighted': overall_profiles["sentence_weighted"],
                            'TSP_word_weighted': overall_profiles["word_weighted"]}
        d = output_file.parent / 'TSP_flow'
        d.mkdir(exist_ok=True, parents=True)
        file = str(output_file.name).replace('_dict.json', '_scores_TSP.json')
        file = d / file
        file = self._handle_windows_path_length_limit(file)
        with open(file, 'w') as f:
            json.dump(overall_TSP_scores, f, indent=4)
        return overall_TSP_scores
        

    def get_meta_scores_from_output(self, output, output_file=None):
        # check '[author info' placeholder in the response
        # get full response from the ANNO file
        anno_file = str(output_file).replace('_dict.json', '_ANNO.json')
        anno_file = Path(self._handle_windows_path_length_limit(anno_file))
        if Path(anno_file).exists():
            response = json.load(open(anno_file, 'r'))['response']
        else:
            anno_file = str(output_file).replace('_dict.json', '_dict_with_input.json')
            anno_file = Path(self._handle_windows_path_length_limit(anno_file))
            response = json.load(open(anno_file, 'r'))['response']
        
        # regex: match '[' optional spaces 'author' optional spaces 'info' optional stuff until ']'
        pattern = re.compile(r"\[\s*author\s+info.*?\]", re.IGNORECASE)
        ph_texts = pattern.findall(response)
        ph_count = len(ph_texts)
        
        questions = output['questions']
        criticisms = output['criticisms']
        requests = output['requests']
        other_responses = output.get('other_responses', [])
        nr_total_items = len(questions) + len(criticisms) + len(requests)
        assert nr_total_items > 0, "No items found in the output."
        
        targeted_items = 0
        for item in questions + criticisms + requests:
             if len(item['response']) != 0:
                 targeted_items += 1
        
       
        item_spec_scores = []
        item_conv_scores = []
        for item in questions + criticisms + requests:
            item_spec_scores.append(item['response_spec_score'])
            item_conv_scores.append(item['response_conv_score']) 

        # get the response text and its length from the ANNO file
        response_word_length = len(response.split())

        scores = {
            'nr_questions': len(questions),
            'nr_criticisms': len(criticisms),
            'nr_requests': len(requests),
            'nr_total_items': nr_total_items,
            'response_word_length': response_word_length,
            'other_responses_sent_count': len(other_responses),
            'other_responses_word_count': sum([len(resp['text'].split()) for resp in other_responses]) if other_responses else 0,
            'other_responses_labels': [resp['labels'] for resp in other_responses] if other_responses else [],
            "ph_count": ph_count,
            "ph_texts": ph_texts
        }

        d = output_file.parent / 'meta'
        d.mkdir(exist_ok=True, parents=True)
        file = str(output_file.name).replace('_dict.json', '_scores_meta.json')
        file = d / file
        with open(file, 'w') as f:
             json.dump(scores, f, indent=4)
        return scores