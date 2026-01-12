from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import json
import pandas as pd
import openai
from openai import OpenAI
import re
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

# ---------------------------
# Utilities
# ---------------------------

def detect_model_family(model_path: str) -> str:
    """
    Very simple heuristic based on the path/name:
    returns 'qwen' | 'llama' | 'other'
    """
    lp = model_path.lower()
    if "qwen" in lp:
        return "qwen"
    if "llama" in lp or "meta-llama" in lp:
        return "llama"
    return "other"

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (Qwen 'thinking' traces)."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

def build_chat_prompt(tokenizer, messages, add_generation_prompt=True,
                      is_qwen=False, enable_reasoning=False):
    """
    Wrap tokenizer.apply_chat_template with Qwen-aware 'enable_thinking'.
    We only pass the flag if explicitly requested and it's Qwen.
    """
    kwargs = dict(
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    if is_qwen and enable_reasoning:
        # Only pass this kwarg for Qwen; HF for LLaMA doesn't accept it.
        kwargs["enable_thinking"] = True
    return tokenizer.apply_chat_template(messages, **kwargs)
    


# ---------------------------
# Loader
# ---------------------------

def load_model_from_path(model_path: str, device_map="auto", four_bit=True):
    """
    Loads models.
    """
    print("Loading model from...", model_path)

    bnb_config = None
    if four_bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Fall back to eos for padding if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=bnb_config,
    )

    family = detect_model_family(model_path)
    print(f"Detected family: {family}")

    return model, tokenizer, family

class Evaluater:
    def __init__(self) -> None:

       ''

    def predict_with_local_model(
        self,
        test,
        output_dir,
        model,
        tokenizer,
        family: str,               # 'qwen' | 'llama' | 'other'
        is_val=False,
        max_tokens=1500, 
        temperature=0,             # 0 for deterministic
        top_p=1.0,                 # use only if temperature > 0, otherwise set to 1.0
        top_k=0,                   # use only if temperature > 0, otherwise set to 0
        enable_reasoning=True,      # used only for Qwen-3
        hide_reasoning=True,        # strip <think> for Qwen-3
        return_full_text=False,     # return prompt+completion (True) or only completion (False)
        output_with_plan=False,     # only useful when self-plan mode is used
    ):
        eval_file = output_dir / ("VAL_eval_pred.csv" if is_val else "eval_pred.csv")
        if eval_file.exists():
            eval_file.unlink()

        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_tokens,
            return_full_text=return_full_text,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id),
        )

        model.eval()

        with torch.inference_mode():
            for i in tqdm(range(len(test)), desc="Generating"):
                print(f"!!!evaluater_Predicting response for {i+1}/{len(test)}: "
                      f"{test[i]['doc_name']}, {test[i]['review_file_id']}, {test[i]['chunk_ix']}")

                system_prompt = test[i]["system_prompt"]
                user_input = test[i]["user_input"] + "\nOutput the response only. Do not include any other text."

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ]

                # Build prompt with family-aware reasoning toggle
                is_qwen = family.lower() == "qwen"
                prompt = build_chat_prompt(
                    tokenizer,
                    messages,
                    add_generation_prompt=True,
                    is_qwen=is_qwen,
                    enable_reasoning=(enable_reasoning if is_qwen else False),
                )

                out = pipe(prompt,
                           temperature=temperature if temperature > 0 else 0.0,
                            do_sample=(temperature > 0),  # ensures False when temp=0
                            top_p=top_p,   # relevant only if temperature > 0
                            top_k=top_k    # relevant only if temperature > 0
                           )
                pred = out[0]["generated_text"]
                pred_with_thinking = pred

                # If Qwen thinking was enabled, optionally strip <think>...</think>
                if is_qwen and hide_reasoning:
                    pred = strip_thinking(pred)

                pred = pred.strip()
                #print('Final response:', pred)
                
                a = self._get_pred_row(test, i, pred, system_prompt, user_input, output_with_plan)
                if enable_reasoning and hide_reasoning and is_qwen:
                        a['orig_output'] = pred_with_thinking
                a.to_csv(eval_file, mode="a", index=False, header=not eval_file.exists())

        return eval_file

    def _get_pred_row(self, test, i, pred, system_prompt, user_input, output_with_plan=False):
                    # get row
                    doc_name = test[i]['doc_name']
                    review_file_id = test[i]['review_file_id']
                    chunk_ix = test[i]['chunk_ix']
                    gold_response = test[i]['gold_response']
                    review_text = test[i]['review_text']
                    if 'user_input_wAIx' in test[i]:
                        user_input_wAIx = test[i]['user_input_wAIx']
                    else:
                        user_input_wAIx = None

                    if output_with_plan:
                        orig_output = pred
                        pred_split = pred.split('###Response:')
                        if len(pred_split) == 2:
                            plan = pred_split[0].strip()
                            new_pred = pred_split[1].strip()
                        else:
                            plan = None
                            new_pred = pred
                        a = pd.DataFrame({"doc_name": [doc_name], "review_file_id": [review_file_id], "chunk_ix": [chunk_ix],
                                    "review_text": [review_text],
                                    "true": [gold_response], "pred": [new_pred], "system_prompt": [system_prompt],
                                    "user_input": [user_input], "user_input_wAIx": [user_input_wAIx],
                                    "plan": [plan], "orig_output": [orig_output]})
                    else:
                        a = pd.DataFrame({"doc_name": [doc_name], "review_file_id": [review_file_id], "chunk_ix": [chunk_ix],
                                    "review_text": [review_text],
                                    "true": [gold_response], "pred": [pred], "system_prompt": [system_prompt],
                                    "user_input": [user_input], "user_input_wAIx": [user_input_wAIx]})
                    return a
    def predict_with_api(self, test, output_dir, api_settings, is_val=False, max_tokens=500, temperature=0, output_with_plan=False):
        eval_file = output_dir / ("VAL_eval_pred.csv" if is_val else "eval_pred.csv")
        if eval_file.exists():
            eval_file.unlink()

        for i in tqdm(range(len(test))):
            
            #print(f"!!!evaluater_Predicting response for {i+1}/{len(test)}: {test[i]['doc_name']}, {test[i]['review_file_id']}, {test[i]['chunk_ix']}")
            system_promt = test[i]["system_prompt"]
            
            user_input = test[i]["user_input"] + "\nOutput the response only. Do not include any other text."
            try:
                if api_settings['api_model_id'].startswith('deepseek'):
                    client = OpenAI(api_key=api_settings['api_key'], base_url=api_settings['api_base'])
                    
                    response = client.chat.completions.create(
                            model="deepseek-reasoner",
                            messages=[
                                {"role": "system", "content": system_promt},
                                {"role": "user", "content": user_input}
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                    cot  = response.choices[0].message.reasoning_content   # CoT (thinking)
                    pred = response.choices[0].message.content             # final answer
                else:
                    #gpt
                    client = openai.AzureOpenAI(api_version=api_settings['api_version'],
                                                azure_endpoint=api_settings['api_base'],
                                                api_key=api_settings['api_key'])
                    
                    response = client.chat.completions.create(
                            model=api_settings['api_model_id'],
                            messages=[
                                {"role": "system", "content": system_promt},
                                {"role": "user", "content": user_input}
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                    pred = response.choices[0].message.content
            except Exception as e:
                print(f"An error occurred: {e}")
                pred = 'none'
            a = self._get_pred_row(test, i, pred, system_promt, user_input, output_with_plan)
            if api_settings['api_model_id'].startswith('deepseek'):
                a['orig_output'] = 'CoT: '+cot+'\nFinal Answer: '+pred
            a.to_csv(eval_file, mode="a", index=False, header=not eval_file.exists())
            
        return eval_file


    def evaluate(self, test, model_path,  output_dir=None,  api_settings=None, 
                do_predict = True, is_val = False, max_tokens=5000, temperature=0,
                eval_gold=False,
                eval_gold_model_name = '',
                eval_pred=True,
                redo_atomic_facts=False, 
                redo_knowledge_source=True,
                output_with_plan=False,
                respeval_eval_types = ['meta','TSP_flow','factuality','conv_spec_direct'],):
        self.max_tokens = max_tokens
        self.model_path = model_path
        start_time = pd.Timestamp.now()

        # do prediction if do_predict is True, otherwise just load the existing eval_pred.csv file
        if do_predict:
            if api_settings is None and model_path is not None:
                model, tokenizer, family = load_model_from_path(model_path, device_map='auto')
                eval_file = self.predict_with_local_model(test, output_dir, model, tokenizer, family, 
                                                          is_val=is_val, max_tokens=max_tokens, 
                                                          temperature=temperature, 
                                                          output_with_plan=output_with_plan)
            
                
            elif api_settings is not None and model_path is None:
                eval_file = self.predict_with_api(test, output_dir, api_settings, 
                                                  is_val=is_val, max_tokens=max_tokens, 
                                                  temperature=temperature, 
                                                  output_with_plan=output_with_plan)
            else:
                raise ValueError('Either api_settings or model_path should be provided.')
            
            end_time = pd.Timestamp.now()
            inference_time = end_time - start_time
            inference_time = inference_time.total_seconds()
            # save the inference time to a json file
            file = output_dir / "generation_cost_time.json"
            # read the existing cost.json file if it exists
            if file.exists():
                with open(file, 'r') as f:
                        cost = json.load(f)
            else:
                cost = {}
            cost['inference_time'] = inference_time
            with open(file, 'w') as f:
                    json.dump(cost, f, indent=4)
        else:
            eval_file = output_dir / "eval_pred.csv"
            inference_time = None
            print(f"Using existing eval file: {eval_file}")


        
        df = pd.read_csv(eval_file)
        # clean the pred and true texts
        df['pred'] = df['pred'].apply(lambda x: x.strip())
        df['true'] = df['true'].apply(lambda x: x.strip())
        
        from tasks.author_response_evaluation.evaluate_author_response import evaluate_author_response
        evaluate_author_response(df, model_path=output_dir, 
                                           eval_gold=eval_gold,
                                           eval_gold_model_name = eval_gold_model_name,
                                           eval_pred=eval_pred,
                                           redo_atomic_facts=redo_atomic_facts, 
                                           redo_knowledge_source=redo_knowledge_source,
                                           eval_types={'basic':None, 'politeness':None, 
                                                       'respeval':respeval_eval_types},) # evaluate basic similarity, politeness, and respeval
    
    