import tiktoken
from pathlib import Path
import pandas as pd
import json
from .data_preprocessor_utils import *
import os


def num_tokens_from_string(string):
    encoding = tiktoken.encoding_for_model('gpt-4o-2024-11-20')
    num_tokens = len(encoding.encode(string))
    return num_tokens

def _handle_windows_path_length_limit(path: Path) -> str:
        # deal with potential Windows path length limitation
        # ensure parent exists (harmless if already exists)
        if isinstance(path, str):
            path = Path(path)
        # convert to absolute and prefix with \\?\  on Windows
        abs_path = path.resolve()
        if os.name == "nt":
            path_for_open = r"\\?\{}".format(str(abs_path))
        else:
            path_for_open = str(abs_path)
        return path_for_open

class DataPreprocessor:
    def __init__(self) -> None:
       print('Preprocessing the data...Gen')

    def preprocess_data(self, dataset, 
                        input_type='inst_nl_icl0',  
                        inst_settings=None,
                        ):
        """
        :param max_length (int): Maximum number of tokens to emit from the tokenizer
        :param input_type (str): Type of format, select from 'inst_nl_iclX-R', 'inst_st_iclX-PN', x is the number of in-context examples can be 1, 2, 3, 4, 5
        """
        self.inst_settings = inst_settings
        self.prompt_st_type = input_type.split('_')[-2]
        
        # Add prompt to each sample
        print("Preprocessing dataset...")
        # create the prompts
        dataset = dataset.map(self.create_prompt_formats, load_from_cache_file=False)
        
        # Shuffle dataset
        seed = 42
        dataset = dataset.shuffle(seed=seed)

        return dataset

    
    def get_content_of_sample(self, sample): 
        # convert certain columns to list
        cols = ['all_edits']
        for col in cols:
            x = sample[col]
            sample[col] = eval(x) if isinstance(x, str) else x
        doc_name = sample['doc_name']
        if 'emnlp24' in doc_name:
            data_path = Path('./data_triplets/emnlp24')
        elif 'peerj' in doc_name:
            data_path = Path('./data_triplets/peerj')
        docs_path = data_path / 'docs' 
        doc_path = docs_path / doc_name
        edits_s = pd.read_csv(doc_path / 'revision' / 'anno' / 'v1-v2_edits_s.csv')   
        edits_p = pd.read_csv(doc_path / 'revision' / 'anno' / 'v1-v2_edits_p.csv')
        edits_s = edits_s.fillna('')
        edits_p = edits_p.fillna('')
        review_file_id = sample['review_file_id']
        response_data = json.load(open(doc_path / 'response' / f"{review_file_id}_response.itg.json", 'r'))
        chunk = [n for n in response_data["response_chunk_nodes_by_quote"] if n['ix'] == sample['chunk_ix']][0]
        # get all_edits as a list of edit_ids
        all_edits = sample['all_edits']
        all_edits_ =[]
        for edit_id in all_edits:
                if '@' in edit_id:
                    edit = edits_s[edits_s['edit_id'] == edit_id].values[0]
                    #convert edit row to dict
                    edit = {col: edit[i] for i, col in enumerate(edits_s.columns)}
                    cols_in_row = [col for col in sample.keys() if 'all_edits_' in col]
                    cols_in_row = [col for col in cols_in_row if 'count' not in col]
                    auto_linking_source = []
                    for col in cols_in_row:
                        if edit_id in sample[col]:
                            auto_linking_source.append(col.replace('all_edits_', ''))
                    edits_p_row = edits_p[edits_p['s_edit_ids'].str.split(';').apply(lambda x: edit_id in x)]
                    assert len(edits_p_row) >= 1, f"Error: {edit_id}"
                    edit['edit_id_P'] = edits_p_row['edit_id'].values[0]  
                    edit['text_src_P'] = edits_p_row['text_src'].values[0]  
                    edit['text_tgt_P'] = edits_p_row['text_tgt'].values[0]  
                    edit['linking_auto'] = auto_linking_source
                    all_edits_.append(edit)
                else:
                    edit = edits_p[edits_p['edit_id'] == edit_id].values[0]
                    #convert edit row to dict
                    edit = {col: edit[i] for i, col in enumerate(edits_p.columns)}
                    cols_in_row = [col for col in sample.keys() if 'all_edits_' in col]
                    cols_in_row = [col for col in cols_in_row if 'count' not in col]
                    auto_linking_source = []
                    for col in cols_in_row:
                        if edit_id in sample[col]:
                            auto_linking_source.append(col.replace('all_edits_', ''))
                    edit['linking_auto'] = auto_linking_source
                    edit['linking_anno_human'] = ''
                    all_edits_.append(edit)
        chunk["all_edits"] = all_edits_  # Updated to assign all_edits_ to chunk["edits"]
        # remove "author_reply_sent_nodes", "meta"
        chunk.pop("author_reply_sent_nodes", None)
        chunk.pop("meta", None)
        return chunk
    
    def get_v1_rag_text(self, sample):
            v1_rag_file = Path('./tasks_data/author_response_generation_prep/selected_samples/v1_RAG_review_top5') / sample['chunk_ix'] / 'v1_RAG_top5.json'
            v1_rag_top5 = json.load(open(v1_rag_file, 'r'))
            v1_rag_text = '- Here are the top 5 paragraphs retrieved from the original paper:\n'
            for r in v1_rag_top5:
                v1_rag_text += f"-- {r['text']}\n"
            return v1_rag_text
    
    def create_test_sample_text(self, sample, inst_settings, edit_type='add'):
        system_prompt_type = inst_settings.get('system_prompt', '')
        style_prompt_type = inst_settings.get('style_prompt', '')
        item_type = inst_settings.get('itemizing', '')
        plan_type = inst_settings.get('planning', '')
        sample_AIx = inst_settings.get('sample_AIx', '')
        length_control = inst_settings.get('length_control', '')
        refining = self.inst_settings.get('refining', '')
        refine_type = refining.get('type', '') if isinstance(refining, dict) else ''
        refine_round = refining.get('round', 0) if isinstance(refining, dict) else 0
        refine_gen = refining.get('refined_gen', '') if isinstance(refining, dict) else ''

        chunk_content = self.get_content_of_sample(sample)
        review_text = chunk_content['quoted_review'] 
        gold_response = chunk_content['author_reply'] 

        promt_st_texts = PROMPT_ST_DIC[self.prompt_st_type]
        review_start = promt_st_texts['review_start']
        review_end = promt_st_texts['review_end']
        
        if system_prompt_type == 'ARR-noAIx':
            if sample_AIx=='':
                sample_text = f"{review_start} {review_text} {review_end}\n\n"
                return sample_text.strip(), review_text.strip(), gold_response.strip()
            elif sample_AIx == 'v1':
                v1_rag_text = self.get_v1_rag_text(sample)
                sample_text = f"{review_start} {review_text} {review_end}\n\n{v1_rag_text}"
                return sample_text.strip(), review_text.strip(), gold_response.strip()
        
        if system_prompt_type == 'ARR-wAIx':
            edits_text = f'- Refer to the author input below: \n'
            added_edit = 0
            for e in chunk_content['all_edits']:
                    #print(e)
                    edit_old = e['text_tgt'] if e['text_tgt'] is not None else ''
                    edit_new = e['text_src'] if e['text_src'] is not None else ''
                    edit_new_P = e['text_src_P'] if e['text_src_P'] is not None else ''
                    edit_old_P = e['text_tgt_P'] if e['text_tgt_P'] is not None else ''
                    edit_action_label = e['ea']
                    edit_intent_label = e['ei']
                    edit_sec_old = e['sec_title_tgt'] if e['sec_title_tgt'] is not None else ''
                    edit_sec_new = e['sec_title_src'] if e['sec_title_src'] is not None else ''
                    
                    if edit_type == 'add':
                        if edit_action_label != 'Add':
                                continue
                        else:
                            if sample_AIx in ['S+SecT+P', 'S+SecT+P+v1']:
                               edit_text = f"-- Authors will add: <{edit_new}> in a paragraph <{edit_new_P}> in Section <{edit_sec_new}>."
                            elif sample_AIx in ['S']:
                               edit_text = f"-- Authors will add: <{edit_new}>."
                    else:
                            if sample_AIx in ['S+SecT+P', 'S+SecT+P+v1']:
                                 if edit_action_label == 'Delete':
                                   edit_text = f"-- Authors will delete: <{edit_old}> from a paragraph <{edit_old_P}> in Section <{edit_sec_old}>."
                                 elif edit_action_label == 'Modify':
                                     edit_text = f"-- Authors will change: <{edit_old}> to <{edit_new}> in a paragraph <{edit_new_P}> in Section <{edit_sec_new}>."
                                 elif edit_action_label == 'Add':
                                       edit_text = f"-- Authors will add: <{edit_new}> in a paragraph <{edit_new_P}> in Section <{edit_sec_new}>."
                            elif sample_AIx in ['S']:
                                    if edit_action_label == 'Delete':
                                       edit_text = f"-- Authors will delete: <{edit_old}>."
                                    elif edit_action_label == 'Modify':
                                        edit_text = f"-- Authors will change: <{edit_old}> to <{edit_new}>."
                                    elif edit_action_label == 'Add':
                                        edit_text = f"-- Authors will add: <{edit_new}>."
                    if edit_text is not None:
                           added_edit += 1
                           edits_text += edit_text + '\n'
            if added_edit==0:
                print(f"Warning: No edits found for chunk {sample['chunk_ix']} in doc {sample['doc_name']}")

            if sample_AIx in ['S+SecT+P+v1']:
                v1_rag_text = self.get_v1_rag_text(sample)
            else:
                v1_rag_text = ''
            
            if item_type == 'item':
                item_file = Path('./tasks_data/author_response_generation_prep/selected_samples/items') / sample['chunk_ix'] / 'items.json'
                items = json.load(open(item_file, 'r'))
                item_input = '  -- The items extracted from the review comment are:\n'
                for k, v in items.items():
                    if k == 'other_responses':
                        continue
                    if isinstance(v, list):
                        if len(v) == 0:
                            item_input += f"   --- {k}: none.\n"
                            continue
                        else: 
                            item_input += f"   --- {k}: "
                            for i, _v in enumerate(v):
                                if isinstance(_v, dict):
                                    review_texts = _v.get('review_text', '')
                                    review_texts = ' '.join(review_texts) if isinstance(review_texts, list) else review_texts
                                    item_input += f"#{i+1}: <{review_texts}> "
                            item_input += '\n'

            else:
                item_input = ''

            if plan_type == 'author-plan':
                plan_file = Path('./tasks_data/author_response_generation_prep/selected_samples/plans') / sample['chunk_ix'] / 'author_plan.json'
                plan_data = json.load(open(plan_file, 'r'))
                plan_text = plan_data.get('plan_text', '')
                if plan_text != '':
                    plan_text = f"- The response action plan is: \n {plan_text}"
            else:
                plan_text = ''
            
            sample_text = f"{review_start} {review_text} {review_end}"
            if item_input:
                sample_text = f"{sample_text}\n{item_input}"
            if plan_text:
                sample_text = f"{sample_text}\n{plan_text}"
            if v1_rag_text:
                sample_text = f"{sample_text}\n\n{v1_rag_text}\n\n{edits_text}"
            else:
                sample_text = f"{sample_text}\n\n{edits_text}"

            # add the previous response and evaluation if it is a refining round
            if refine_type != '' and refine_gen != '':
                refined_gen_file = Path('./results/author_response_generation/inference_llm') / refine_gen / 'eval_pred.csv'
                refined_gen_file = Path(_handle_windows_path_length_limit(refined_gen_file))
                refined_gen_df = pd.read_csv(refined_gen_file)
                refined_gen_row = refined_gen_df[(refined_gen_df['chunk_ix'] == sample['chunk_ix'])]
                refined_gen_response = refined_gen_row['pred'].values[0].strip() if len(refined_gen_row) > 0 else ''
                refine_text = f"- The previous response generated is: \n {refined_gen_response}\n\n"

                if refine_type.startswith('refine-quality'):

                    chunk_ix = sample['chunk_ix']
                    refined_gen_eval_file = Path('./.cache/respscore/eval_results') / refine_gen / chunk_ix / 'pred' / 'conv_spec_direct' / '@pred_conv_spec_direct.json'
                    refined_gen_eval_file = Path(_handle_windows_path_length_limit(refined_gen_eval_file))
                    
                    if refined_gen_eval_file.exists():
                        with open(refined_gen_eval_file, 'r') as f:
                            refined_gen_eval = json.load(f)
                        resp_overall = refined_gen_eval.get('overall', '')
                    # convert dict to text
                    resp_overall = json.dumps(resp_overall, indent=2) if isinstance(resp_overall, dict) else resp_overall
                    refine_text+= f"- The overall response scores (directness, specificity and convincingness, 5-point scale) and the respective justifications and improvement suggestions:\n {resp_overall}\n"
                    task_text = "TASK: Please revise the previous response based on the review comment, the provided inputs and the requirements, as well as the evaluation results above to improve the directness, specificity and convincingness scores. Output the revised response only.\n"
                
                if refine_type == 'refine-quality-fact':
                    chunk_ix = sample['chunk_ix']
                    refined_gen_eval_file = Path('./.cache/respscore/eval_results') / refine_gen / chunk_ix / 'pred' / 'factuality' / '@pred_scores_RAG+GPT.json'
                    refined_gen_eval_file = Path(_handle_windows_path_length_limit(refined_gen_eval_file))
                    
                    if refined_gen_eval_file.exists():
                        with open(refined_gen_eval_file, 'r') as f:
                            refined_gen_eval = json.load(f)
                        
                        fact_score = refined_gen_eval.get('user-input', '')
                        fact_score = fact_score.get('score', 0)
                        fact_score = round(fact_score * 100, 2)
                        
                        # convert dict to text
                        fact_text = f'- Factuality score: {fact_score}% of the atomic facts in the previous response are supported by the provided inputs.'
                        refine_text+= f"\n{fact_text}"
                        task_text = "TASK: Please revise the previous response based on the review comment, the provided inputs and the requirements, as well as the evaluation results above to improve the directness, specificity, convincingness and the factuality of the response. Output the revised response only.\n"
                    
                refine_text+= f"\n\n{task_text}"
                sample_text = f"{sample_text}\n\n{refine_text}"
                
            return sample_text.strip(), review_text.strip(), gold_response.strip()

            
    def create_prompt_formats(self, sample):
        """
        Creates a formatted prompt template for a prompt in the dataset
        :param sample: sample from the dataset
        """
        # create the system prompt - task description + style prompt to distinguish ARR vs. Journal
        
        # get task description style prompt
        system_prompt_type = self.inst_settings.get('system_prompt', '')
        style_prompt_type = self.inst_settings.get('style_prompt', '')
        item_type = self.inst_settings.get('itemizing', '')
        plan_type = self.inst_settings.get('planning', '')
        sample_AIx_type = self.inst_settings.get('sample_AIx', '')
        length_control = self.inst_settings.get('length_control', '')
        refining = self.inst_settings.get('refining', '')
        refine_type = refining.get('type', '') if isinstance(refining, dict) else ''
        refine_round = refining.get('round', 0) if isinstance(refining, dict) else 0
        refine_gen = refining.get('refined_gen', '') if isinstance(refining, dict) else ''


        if system_prompt_type == 'ARR-noAIx' and item_type == '':
            task_description = task_description_noAIx
        elif system_prompt_type == 'ARR-wAIx' and item_type == '':
            task_description = task_description_wAIx_ARR
        elif system_prompt_type == 'ARR-noAIx' and item_type == 'item':
            task_description = task_description_noAIx_item
        elif system_prompt_type == 'ARR-wAIx' and item_type == 'item':
            task_description = task_description_wAIx_ARR_item
            if plan_type == 'author-plan':
                task_description = task_description + '\n' + task_description_wAIx_ARR_item_authorplan
        else:
            raise ValueError(f"Unknown system prompt type: {system_prompt_type}")
        
        if refine_type != '' and refine_gen != '':
            task_description = task_description + f"\nNote: This is a refinement round to improve the quality of the previous generated response based on its evaluation results."

        #get style prompt
        if style_prompt_type == 'style':
            if system_prompt_type == 'ARR-noAIx':
                style_prompt = style_prompt_noAIx_ARR
            elif system_prompt_type == 'ARR-wAIx':
                style_prompt = style_prompt_wAIx_ARR
            else:
                raise ValueError(f"Unknown style prompt type: {style_prompt_type}")
        elif style_prompt_type == 'style-PH':
            if system_prompt_type == 'ARR-noAIx':
                style_prompt = style_prompt_noAIx_ARR_PH
            elif system_prompt_type == 'ARR-wAIx':
                style_prompt = style_prompt_wAIx_ARR_PH
            else:
                raise ValueError(f"Unknown style prompt type: {style_prompt_type}")
            
        system_prompt = [task_description, style_prompt]
        system_prompt = [i for i in system_prompt if i]  # remove empty strings
        system_prompt = '\n'.join(system_prompt)

        # create the user input - test sample converted to the same format
        user_input, review_text, gold_response = self.create_test_sample_text(sample, self.inst_settings)


        # if 'noAIx' , create an additional column 'user_input_wAIx' that contains the user input with edits, for evaluation purposes
        if 'noAIx' in system_prompt_type:
            new_system_prompt_type = system_prompt_type.replace('noAIx', 'wAIx')
            if 'v1' in sample_AIx_type:
                new_sample_AIx_type = sample_AIx_type.replace('v1', 'S+SecT+P+v1')
            else:
                new_sample_AIx_type = 'S+SecT+P'

            new_inst_settings = {
                'system_prompt': new_system_prompt_type,
                'style_prompt': style_prompt_type,
                'itemizing': item_type,
                'planning': plan_type,
                'sample_AIx': new_sample_AIx_type,
                'length_control': length_control,
                'refining': refining
            }

            user_input_wAIx, _, _ = self.create_test_sample_text(sample, new_inst_settings)
            sample['user_input_wAIx'] = user_input_wAIx
        
        if length_control != '':
            if length_control.startswith('dyn-upper-n'):
                n = int(length_control.split('n+')[1])
                max_len = len(gold_response.strip().split()) + n
                length_control_text = f"Please limit the response to NO MORE than {max_len} words."
            else:
                raise ValueError(f"Unknown length control type: {length_control}")
            system_prompt = f"{system_prompt}\n{length_control_text}"

        sample['system_prompt'] = system_prompt
        sample['user_input'] = user_input
        sample['review_text'] = review_text
        sample['gold_response'] = gold_response

        total_input_tokens = num_tokens_from_string(system_prompt) + num_tokens_from_string(user_input)
        sample['total_input_tokens'] = total_input_tokens
        return sample
    