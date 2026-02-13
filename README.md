# Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review 
This is the official code repository for the paper "Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review", presented at XXX conference. It contains the scripts for author response generation and evaluation outlined in the paper.

Please find the paper [here](https://arxiv.org/abs/2602.11173), and star the repository to stay updated with the latest information.

In case of questions please contact [Qian Ruan](mailto:ruan@ukp.tu-darmstadt.de).

## Abstract
Author response (rebuttal) writing is a critical stage of scientific peer review that demands substantial author effort. Recent work frames this task as automatic text generation, underusing author expertise and intent. 
In practice, authors possess domain expertise, author-only information, revision and response strategies--concrete forms of author expertise and intent--to address reviewer concerns, and seek NLP assistance that integrates these signals to support effective response writing in peer review.
We reformulate author response generation as an author-in-the-loop task and introduce *REspGen*, a generation framework that integrates explicit author input, multi-attribute control, and evaluation-guided refinement, together with *REspEval*, a comprehensive evaluation suite with 20+ metrics covering input utilization, controllability, response quality, and discourse. To support this formulation, we construct *Re<sup>3</sup>Align*, the first large-scale dataset of aligned review--response--revision triplets, where revisions provide signals of author expertise and intent.
Experiments with state-of-the-art LLMs show the benefits of author input and evaluation-guided refinement, the impact of input design on response quality, and trade-offs between controllability and quality. We make our dataset, generation and evaluation tools publicly available.

## Frameworks and Dataset
![](/resource/fig1.png)

*Figure 1. In this work, we contribute (1) REspGen, an author-in-the-loop ARG framework that integrates explicit author input (d), controllable planning and length (b–c), and additional paper context (e);
(2) Re<sup>3</sup>Align, the first large-scale review–response–revision triplets dataset for modeling author signals; and
(3) REspEval, a comprehensive response evaluation framework with over 20 metrics spanning four dimensions.*

## Quickstart
1. Download the project from github.
```bash
git clone https://github.com/UKPLab/arxiv2026-respgen-respeval
```

2. Setup environment
```bash
python -m venv .arxiv2026-respgen-respeval
source ./.arxiv2026-respgen-respeval/bin/activate
pip install -r requirements.txt
```   
   
## Data 
Download the *Re<sup>3</sup>Align* dataset from [1] and extract the subfolders to ./data_triplets and ./tasks_data. 

[1]. https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4982 

### Author Response Generation and Evaluation
Check the 'generate_evaluate_author_response.py' script for the complete pipeline and alternative settings. You can customize the arguments within \<settings\> and \</settings\>. 

For example, author response generation with GPT-4o under REspGen-Setting 6 (see the paper):

1. Basic Settings

```python
# basic settings
# <settings>
task_name ='author_response_generation'
method = 'inference_llm' 
data_root_path = 'tasks_data' # root path of the triplet linking data
train_type = None # name of the training data in data/{task_name}, none for inference mode
val_type = None # name of the validation data in data/{task_name}, none for inference mode
test_type = 'selected_samples' # name of the test data in data/{task_name}
```
2. Load Data

```python
from tasks.task_data_loader import TaskDataLoader
task_data_loader = TaskDataLoader(data_root=data_root_path, task_name=task_name, test_type=test_type, train_type=train_type, val_type=val_type)
train_ds, val_ds, test_ds= task_data_loader.load_data()
```

3. Specify Model and Experimental Settings

```python
# <settings>
llm_model_name = 'gpt-4o-2024-11-20' 
# path to saved api keys
api_key_path_dict = {'deepseek-r1': '.keys/deepseek_key.txt',
                 'gpt-4o-2024-11-20': '.keys/azure_key.txt',}
# generation settings
input_type = 'inst_nl_icl0'  # natural language instructions with 0 in-context learning examples
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
```
4. Preprocess Data

```python
from tasks.task_data_preprocessor import TaskDataPreprocessor
data_preprocessor = TaskDataPreprocessor(task_name=task_name, method=method).data_preprocessor
test_ds = data_preprocessor.preprocess_data(test_ds, 
                                                input_type=input_type, 
                                                inst_settings=inst_settings, 
                                                )
```
5. Create a model folder to save generations

```python
# create output dir
# <settings>
recreate_dir = True # Create a directory for the model, true: recreate and rerun generation and evaluation if exists, false: not recreate if exists
# </settings>
# create model dir under ./results to save the generated outputs
output_dir = create_model_dir(task_name, method, llm_model_name, train_type, test_type, input_type, inst_type,  recreate_dir=recreate_dir)
```
6. Generate and Evaluate Author Response

```python
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
# create generator and evaluator
from tasks.task_evaluater import TaskEvaluater
do_predict = True #generate responses
eval_gold = False # do not evaluate human gold responses, this should be set for true once at the beginning to get the gold eval results
eval_gold_model_name = 'gpt-4o-2024-11-20_None_selected_samples_inst_nl_icl0_@ARR-noAIx_style-PH' # model name where the gold responses were evaluated, needed if eval_gold is TFalse
eval_pred = True # evaluate the model-generated responses
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

evaluater.evaluate(test_ds, 
                       output_dir=output_dir, 
                       model_path=local_model_path,
                       api_settings=api_settings, 
                       do_predict=do_predict, 
                       eval_gold=eval_gold,
                       eval_gold_model_name = eval_gold_model_name,
                       eval_pred=eval_pred,
                       respeval_eval_types=respeval_eval_types)
# The evaluation reports are saved as JSON files in the model folder under ./results, including basic similarity-based, politeness,  and REspEval scores 
```

## Citation

Please use the following citation:

Ruan, Q., & Gurevych, I. (2026). Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review. ArXiv. https://arxiv.org/abs/2602.11173 
```
@misc{ruan2026authorintheloopresponsegenerationevaluation,
      title={Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review}, 
      author={Qian Ruan and Iryna Gurevych},
      year={2026},
      eprint={2602.11173},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.11173}, 
}
```

## Disclaimer
This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

<https://intertext.ukp-lab.de/>

<https://www.ukp.tu-darmstadt.de>

<https://www.tu-darmstadt.de>
