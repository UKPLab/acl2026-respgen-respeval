import json
from REspEval.respeval.openai_lm import OpenAIModel
import logging, sys
from pathlib import Path
from REspEval.respeval.utils_TSP_flow_aggregate_plot import _handle_windows_path_length_limit
from REspEval.respeval.utils_json_output_process import load_json_robust

logging.basicConfig(
    level=logging.INFO,                      # show INFO and above
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,                       # send to stdout (default is stderr)
    force=True                               # override existing handlers (Py>=3.8)
)

class AtomicFactGenerator(object):
    def __init__(self, key_path, model_name="gpt-5", output_file=None, cache_dir=".cache/respeval/cache_files"):
        output_file = Path(_handle_windows_path_length_limit(Path(output_file)))
        self.output_file = output_file
        self.model_name = model_name
        self.cache_dir = cache_dir
        cache_file = Path(self.cache_dir) / f"{self.model_name}_atomic_facts.db"
        self.openai_lm = OpenAIModel(model_name, key_path=key_path, cache_file=cache_file)

    def run(self, generation_texts_list, review):
        """Convert the generation into a set of atomic facts. Return a total words cost if cost_estimate != None."""
        assert isinstance(generation_texts_list, list), "generation must be a list of strings, can be sentences or paragraphs."
        logging.info(f"atomic facts: getting atomic facts")
        
        return self.get_atomic_facts_from_texts(generation_texts_list, review)
    
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
        logging.info("atomic facts: Estimated OpenAI API cost for %s : $%.6f for %d input words and %d output words" % (task,  total_cost, input_words, output_words))
        return total_cost
    
    def get_atomic_facts_from_texts(self, texts, review):
       
        system_prompt = """Your task is to read the given texts (author response) and extact atomic facts from them. 
        The review comment is also provided for context, but do not include it in the output.
        Output a JSON object containing a list of dictionaries from the given list of texts. 
        Each dictionary must have two keys: "text" and "facts".
       "text": the original sentence or passage from the input.
       "facts": a list of minimal, independent facts derived from the corresponding "text".

        Guidelines:
        - Split conjunctive or compound statements into separate facts.
        - Exclude hedges, intentions, social niceties, structural notes, and meta-commentary (e.g., "we hope," "we will update," "thank you").
        - Preserve as many facts as possible.
        - Retain numbers, dataset names, model/component names, and all concrete technical claims.
        - Return only JSON.

        Below an example output:
        [
  {
    "text": "MEI is the task of identifying mentions that refer to specified entities of interest.",
    "facts": [
      "MEI is the task of identifying mentions that refer to specified entities of interest."
    ]
  },
  {
    "text": "There is no explicit restriction for these specified entities to be exactly the set of 'most frequent' entities.",
    "facts": [
      "There is no explicit restriction for specified entities in MEI to be the set of most frequent entities."
    ]
  },
  {
    "text": "MEI assumes that the user has access to entity information and aims to track these entities.",
    "facts": [
      "MEI assumes that the user has access to entity information.",
      "MEI aims to track specified entities."
    ]
  },
  {
    "text": "The MovieCoref dataset explicitly provides the representative phrases as additional annotation, and we directly utilize them in our experiments (a prerun of CR is not required).",
    "facts": [
      "The MovieCoref dataset explicitly provides representative phrases as additional annotation.",
      "Representative phrases from the MovieCoref dataset are directly utilized in experiments.",
      "A prerun of CR is not required when using the MovieCoref dataset."
    ]
  },
  {
    "text": "Applications could be tracking interesting entities in long discourses like novels and movie scripts.",
    "facts": [
      "Applications of MEI include tracking interesting entities in novels.",
      "Applications of MEI include tracking interesting entities in movie scripts."
    ]
  },
  {
    "text": "Information about entities is readily available on websites like IMDb for movies and SparkNotes for novels.",
    "facts": [
      "Information about movie entities is available on IMDb.",
      "Information about novel entities is available on SparkNotes."
    ]
  },
  {
    "text": "Such information is already being used as part of various benchmarks [4] [5].",
    "facts": [
      "Entity information from sources like IMDb and SparkNotes is used as part of various benchmarks."
    ]
  },
  {
    "text": "The applicability of MEI depends on the availability of such external meta-data/user’s own knowledge.",
    "facts": [
      "The applicability of MEI depends on the availability of external meta-data.",
      "The applicability of MEI depends on the user’s own knowledge."
    ]
  }
]

       """
        user_input = f"review comment: {review}\n\n"  + f"Input texts: {texts}\n\n" 
        
        output = self.openai_lm.generate(system_prompt, user_input, max_output_length=1500)
        # save raw output for debugging in case not valid JSON
        output_file2 = str(self.output_file).replace('.json', '_B.json')
        output_file2 = Path(_handle_windows_path_length_limit(Path(output_file2)))
        with open(output_file2, 'w') as f:
                json.dump(output, f, indent=4)
        output = load_json_robust(output)
        with open(self.output_file, 'w') as f:
                json.dump(output, f, indent=4)

        input_words = len(system_prompt.split()) + len(user_input.split())
        output_words = len(str(output).split())
        total_cost = self.cost_estimates(input_words, output_words, task="atomic fact extraction", model=self.model_name)
                    
        return output, total_cost