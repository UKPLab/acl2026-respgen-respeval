from pathlib import Path
import shutil
import pandas as pd

def create_model_dir(task_name, method, model_dic_key, train_type, test_type,  input_type, inst_type, recreate_dir=True, max_length=None):
    # create a model dir (unique identifier) to save the generated outputs
    output_dir = Path("./results")
    if not output_dir.exists():
            output_dir.mkdir()
    output_dir = output_dir/task_name
    if not output_dir.exists():
            output_dir.mkdir()
    output_dir = output_dir/method
    if not output_dir.exists():
            output_dir.mkdir()
    model_folder_name = model_dic_key
    if max_length is not None:
       model_folder_name += f'_{train_type}_{test_type}_ml{max_length}_{input_type}_{inst_type}'
    else:
       model_folder_name += f'_{train_type}_{test_type}_{input_type}_{inst_type}'
    output_dir = Path(output_dir/model_folder_name)
    if output_dir.exists():
         if recreate_dir:
            shutil.rmtree(output_dir)
            output_dir.mkdir()
    else:
        output_dir.mkdir()
    return output_dir
     
def main():
    ############################################################################
    # basic settings
    # <settings>
    task_name ='author_response_generation'
    method = 'inference_llm' 
    data_root_path = 'tasks_data' # root path of the triplet linking data
    train_type = None # name of the training data in data/{task_name}, none for inference mode
    val_type = None # name of the validation data in data/{task_name}, none for inference mode
    test_type = 'selected_samples' # name of the test data in data/{task_name}
    # </settings>
    print('========== Basic settings: ==========')
    print(f'task_name: {task_name}')
    print(f'method: {method}')
    print(f'test_type: {test_type}')
    ############################################################################
    # load task data
    from tasks.task_data_loader import TaskDataLoader
    task_data_loader = TaskDataLoader(data_root=data_root_path, task_name=task_name, test_type=test_type, train_type=train_type, val_type=val_type)
    train_ds, val_ds, test_ds= task_data_loader.load_data()
    print('========== 1. Task data loaded: ==========')
    print(f'train_ds: {train_ds}')
    print(f'val_ds: {val_ds}')
    print(f'test_ds: {test_ds}')
    
    ############################################################################
    # preprocess data from generation
    # <settings>
    llm_model_name = 'gpt-4o-2024-11-20' 
    api_key_path_dict = {'deepseek-r1': '.keys/deepseek_key.txt',
                         'gpt-4o-2024-11-20': '.keys/azure_key.txt',}
    # generation settings
    input_type = 'inst_nl_icl0'  # natural language instructions with 0 in-context learning examples, other settings can be added later
    
    '''
    # definitions of the instruction settings
    style_prompts = ['style', 'style-PH'] # style-PH: insert placeholder if author-only information is needed, style: general style prompt
    system_prompts = ['ARR-noAIx', 'ARR-wAIx'] # ARR-noAIx: without author input, ARR-wAIx: with author input
    sample_AIxs = ['S',  'S+SecT+P',  'S+SecT+P+v1'] # S: author input as edit strings only, S+SecT+P: author input as edit strings + paragraph context + section title, S+SecT+P+v1: author input as edit strings + paragraph context + section title + v1 retrieval
    itemizing_list = ['','item'] # '': no itemizing, 'item': itemize the review, 'item' is default if response plan control is used
    planning_list = ['','author-plan'] # '': no planning control,  'author-plan': author control over response plan
    length_control = ['','dyn-upper-n+50'] # '': no length control, 'dyn-upper-n+50': author control over response length, defined as dynamic upper bound n+50
    # an example setting with refining step
    refining = {'type': 'refine-quality-fact', # refine-quality-fact: refine for better quality and factuality
                'round': 1,
                'refined_gen': 'deepseek-r1_None_selected_samples_inst_nl_icl0_@ARR-wAIx_style-PH_S+SecT+P+v1_item_author-plan_dyn-upper-n+50_temp0'} #refine the outputs from this folder in 'results/author_response_generation/inference_llm/' 
    refining_text = f"{refining['type']}-R{refining['round']}"
    '''
    # an example setting (see Setting 6 in the paper) 
    system_prompt = 'ARR-wAIx'# with author input
    style_prompt = 'style-PH' # insert placeholder if author-only information is needed
    sample_AIx = 'S+SecT+P+v1'# author input as edit strings + paragraph context + section title + v1 retrieval
    itemizing = 'item' # itemize the review
    planning = 'author-plan'# author control over response plan
    length_control = 'dyn-upper-n+50'# author control over response length
    refining = {} # not a refining step
    refining_text = '' # not a refining step
    # </settings>

    inst_settings = {'system_prompt': system_prompt, 
                          'style_prompt': style_prompt, 
                          'sample_AIx': sample_AIx, 
                          'itemizing': itemizing, 
                          'planning': planning,
                          'length_control': length_control,
                          'refining': refining
                          }
    inst_settings_texts = {'system_prompt': system_prompt, 
                          'style_prompt': style_prompt, 
                          'sample_AIx': sample_AIx, 
                          'itemizing': itemizing, 
                          'planning': planning,
                          'length_control': length_control,
                          'refining': refining_text
                          }
    inst_type = [v for k, v in inst_settings_texts.items() if v != '']
    inst_type = '_'.join(inst_type)
    inst_type = '@'+ inst_type 
    
    print('=====Creating test samples with the following settings:')
    print(f"  system_prompt: {system_prompt}")
    print(f"  style_prompt: {style_prompt}")
    print(f"  sample_AIx: {sample_AIx}")
    print(f"  itemizing: {itemizing}")
    print(f"  planning: {planning}")
    print(f"  length_control: {length_control}")
    print(f"  refining: {refining}")
    
    from tasks.task_data_preprocessor import TaskDataPreprocessor
    data_preprocessor = TaskDataPreprocessor(task_name=task_name, method=method).data_preprocessor

    test_ds = data_preprocessor.preprocess_data(test_ds, 
                                                input_type=input_type, 
                                                inst_settings=inst_settings, 
                                                )
    
    print('========== 3. Dataset preprocessed: ==========')
    print('test_ds: ', test_ds)
    print(test_ds[0])
    
    ############################################################################
    # create output dir
    # <settings>
    recreate_dir = True # Create a directory for the model, true: recreate and rerun generation and evaluation if exists, false: not recreate if exists
    # </settings>
    # create model dir to save the generated outputs
    output_dir = create_model_dir(task_name, method, llm_model_name, train_type, test_type, input_type, inst_type,  recreate_dir=recreate_dir)
    print('========== 4. Model dir created: ==========')
    print('output_dir: ', output_dir)

    # specify the local model name and path if using local models like LLaMA, Qwen, etc.
    local_model_name_path_dict = {
    'llama-3.3-70b-inst': '',
    'qwen3-32b': '',
    'phi-4-reasoning': '',
     }
    local_model_path = local_model_name_path_dict[llm_model_name] if llm_model_name in local_model_name_path_dict else None
    
    #update api_settings if not using local model
    if local_model_path is None:
        assert llm_model_name in api_key_path_dict, f'Please provide the key path for the model {llm_model_name} in key_path_dict'
        key_path = api_key_path_dict[llm_model_name]
        # read api key and base from the key file
        with open(key_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            api_version = lines[0].strip().split('=')[1].strip()
            api_base    = lines[1].strip().split('=')[1].strip()
            api_key     = lines[2].strip().split('=')[1].strip()

            api_settings = {'api_version': api_version, # empty string '' if not needed
                            'api_base': api_base, 
                            'api_key': api_key, 
                            'api_model_id': llm_model_name
                            }
    else:
        api_settings = None
    
    ############################################################################
    # create generator and evaluater
    from tasks.task_evaluater import TaskEvaluater
    do_predict = True #generate responses
    eval_gold = False # donnot evluate human gold responses, this should be set for true once at the beginning to get the gold eval results
    eval_gold_model_name = 'gpt-4o-2024-11-20_None_selected_samples_inst_nl_icl0_@ARR-noAIx_style-PH' # model name where the gold responses were evaluated, needed if eval_gold is TFalse
    eval_pred = True # evaluate the model generated responses
    evaluater = TaskEvaluater(task_name=task_name, method=method).evaluater
    # define the metrics to evaluate
    if inst_settings['length_control'].strip()!='':
            respeval_eval_types = ['meta','TSP_flow','factuality','conv_spec_direct', 'len_control']
    else:
            respeval_eval_types = ['meta','TSP_flow','factuality','conv_spec_direct']

    if inst_settings['planning'].strip()!='':
            respeval_eval_types.append('plan')
    if itemizing == '' and planning == '' and length_control == '' and refining == {}:
            respeval_eval_types.append('ICR')

    print('========== 5. Evaluating the model: ==========')
    print('do_predict: ', do_predict)
    print('eval_gold: ', eval_gold)
    print('eval_gold_model_name: ', eval_gold_model_name)
    print('eval_pred: ', eval_pred)
    print('respeval_eval_types: ', respeval_eval_types)

    evaluater.evaluate(test_ds, 
                       output_dir=output_dir, 
                       model_path=local_model_path,
                       api_settings=api_settings, 
                       do_predict=do_predict, 
                       eval_gold=eval_gold,
                       eval_gold_model_name = eval_gold_model_name,
                       eval_pred=eval_pred,
                       respeval_eval_types=respeval_eval_types)

    print('========== 6. Model evaluated ==========')
    print('output_dir: ', output_dir)
    print('========== DONE ==========')
    

if __name__ == "__main__":
    main()
