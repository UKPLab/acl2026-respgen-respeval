
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from REspEval.respeval.openai_lm import OpenAIModel

import logging, sys

logging.basicConfig(
    level=logging.INFO,                      # show INFO and above
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,                       # send to stdout (default is stderr)
    force=True                               # override existing handlers (Py>=3.8)
)

SYSTEM_PROMPT = """
You are an impartial LLM judge. Your input is a JSON object with keys:
- "review comment": string (the reviewer’s comment)
- "response": string (the author’s reply to the comment)
- "paired items": array of objects, each with:
    - "id": string (e.g., "criticism_1"), the item type can be "question", "criticism", or "request"
    - "review_texts": array of strings
    - "response_spans": array of strings that the upstream system believes address the item
- "not linked response spans": array of strings (response parts not linked to any item)
    - "id": string (e.g., "unlinked_1")
    - "response_spans": string

Your task: produce a SINGLE overall evaluation (no per-item scores) of the author’s response along three axes:
A) Directness — how clearly the response targets and engages with the reviewer’s item(s).
B) Specificity — how much concrete detail and precision the response provides.
C) Convincingness — how persuasive and well-justified the response is.


Use the full "review comment" and "response" for context, but ground your reasoning primarily in the provided "response_spans" when they are relevant. Do NOT invent facts, numbers, or references that are not present. If you reference an item in your justifications or suggestions, include its item id in square brackets, e.g., "[criticism_1]".

---

SCORING RUBRICS (integers 1–5 only):
A. Directness (targeting & alignment with reviewer’s item)
- 5 – Very direct: Explicitly and fully engages with the reviewer’s item; clear alignment between review comment and response.
- 4 – Direct: Addresses the item clearly, though may wander slightly or partially broaden the scope.
- 3 – Partly direct: Some engagement with the item, but diluted or mixed with unrelated content.
- 2 – Weakly direct: Minimal or tangential engagement with the item; mostly off-topic.
- 1 – Not direct: Does not engage with the reviewer’s item at all.

B. Convincingness (persuasiveness & justification quality)
- 5 – Very convincing: Directly resolves the concern(s) with strong evidence (data, math, citations, explicit section/table/figure references) and clear logic; anticipates counterpoints where relevant.
- 4 – Strong: Substantively addresses the concern(s) with clear reasoning and at least one concrete support (e.g., section/table reference or quantitative detail). Minor gaps remain.
- 3 – Moderate: Engages the point(s) and offers some reasoning, but support is partial, qualitative, or incomplete; notable uncertainties remain.
- 2 – Weak: Acknowledges the point(s) but relies on assertion or vague justification; little to no concrete support.
- 1 – Not convincing: Ignores/deflects or contradicts without support; non-responsive or purely social niceties.

C. Specificity (precision & concreteness of detail)
- 5 – Very specific: Rich in precise details such as numbers, datasets, metrics, configurations, ablations, implementation details, and explicit section/table/figure pointers.
- 4 – High: Multiple concrete details (named components, explicit comparisons, at least one clear reference); some fine-grained details may be missing.
- 3 – Moderate: Some specific elements (e.g., naming components or methods) but limited detail; few or no numbers/references; scope partly vague.
- 2 – Low: Mostly general statements; promises to “clarify” without specifying where/how.
- 1 – Very vague: Generic acknowledgments; no concrete or actionable detail.



---

OUTPUT REQUIREMENTS:
- Return ONLY valid JSON matching the schema below. No extra prose, no backticks.
- Scores must be integers in [1, 5]. Do NOT output floats or 0.
- Justifications are the reasoning of the scores. + for strengths and - for weaknesses. Keep justifications concise (bullet-like strings), tie them to concrete evidence in the response/response_spans, and include ids when relevant.
- Suggestions must be actionable steps (1-2 per metric) that, if implemented, would plausibly raise the score to 5.
---

OUTPUT SCHEMA:
{
  "overall": {
    "directness": 1,
    "specificity": 1,
    "convincingness": 1,
    "justifications": {
      "directness": ["short bullet-like reasons, start with + or -"],
      "specificity": ["short bullet-like reasons, start with + or -"],
      "convincingness": ["short bullet-like reasons, start with + or -"],
    },
    
    "improve_suggestions_to_5":  {
      "directness": ["1-2 actionable suggestions to improve targeting"],
      "specificity": ["1-2 actionable suggestions to improve details"],
      "convincingness": ["1-2 actionable suggestions to improve persuasiveness"],
    },
      
  },
  "meta": {
    "confidence": 0.0,
    "judge_notes": "optional short note"
  }
}

FIELD RULES:
- "confidence" is a float in [0,1] reflecting your certainty in the overall assessment.
- If some paired items are not addressed by any meaningful span, reflect this in lower scores and mention their ids in justifications (e.g., "No direct engagement with [criticism_2]").
- Do not fabricate section/table/figure numbers. Only cite what appears in the response; if absent, penalize appropriately per rubric.

Produce only the JSON object described above.
"""


class ConvSpecDirectScorer:
    """
   
    """

    def __init__(
        self,
        data_dir: str = "",
        openai_key: str = "api.key",
        openai_model: str = "gpt-5",
        cache_dir: str = ".cache/respeval/cache_files",
    ):
        self.data_dir = Path(data_dir)
        # LLM (only for GPT mode)
        self.openai_model = openai_model
        self.lm = OpenAIModel(
            openai_model,
            key_path=openai_key,
            cache_file=Path(cache_dir) / f"{openai_model}_quality_scores.db",
        ) 

    def get_user_input(self, review_text, response_text, linked_data):
        def get_paired_items(linked_data):
            paired_items=[]
            for section in  ['questions', 'criticisms', 'requests']:
                section_item_id = 0
                section_items = linked_data.get(section, [])
                #print("!!!section, section_items", section, section_items)
                if section_items:
                    for item in section_items:
                        item_id = section_item_id + 1
                        item_id_str = f"{section[:-1]}_{item_id}"
                        review_texts = item.get('review_text', [])
                        response_spans = item.get('response', [])
                        response_spans = [s['text'].strip() for s in response_spans]
                        paired_items.append({'id': item_id_str, 'review_texts': review_texts, 'response_spans': response_spans})
            not_linked = linked_data.get('other_responses', [])
            not_linked = [{"id": f"unlinked_{i+1}", "response_spans": s['text'].strip()} for i,s in enumerate(not_linked)]
            
            return paired_items, not_linked
        if linked_data is None:
            paired_items = None
            not_linked = None
        else:
            paired_items, not_linked = get_paired_items(linked_data)
        user_input_dict = {
            "review comment": review_text,
            "response": response_text,
            "paired items": paired_items,
            "not linked response spans": not_linked
        }
        return user_input_dict
        
    def cost_estimates(self, input_words, output_words, task='', model=''):
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


    # ---------- Scoring ----------
    def get_score(
        self,
        linked_data: Dict[str, Any],
        review_text: str,
        response_text:str,
    ) -> Dict[str, Any]:
        
        user_input_dict = self.get_user_input(review_text, response_text, linked_data)
        user_input = json.dumps(user_input_dict, indent=2)
        user_input = f"The input json is:\n{user_input}\n\nOuput json:\n\n"
        
        system_prompt = SYSTEM_PROMPT.strip()
        prompt = system_prompt + '\n\n' +user_input
        input_words = len(prompt.split())
        out = self.lm.generate(system_prompt=system_prompt, user_input=user_input, max_output_length=8000)
        ans = out[0] if isinstance(out, tuple) else out
        output_words = len(ans.split())

        total_cost = self.cost_estimates(input_words, output_words, model=self.openai_model, task="quality scoring")
        # convert ans to json
        try:
            ans = json.loads(ans)#
            ans['meta']['input_words'] = input_words
            ans['meta']['output_words'] = output_words
            ans['meta']['total_cost'] = total_cost
            ans['meta']['model'] = self.openai_model
            ans['user_input'] = user_input_dict
        except:
            print("!!!conv_spec_scorer.py_Error in parsing the output to json, returning empty dict")
            ans = str(ans)

        return ans
