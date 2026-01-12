
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from REspEval.respeval.conv_spec_direct_scorer import ConvSpecDirectScorer
from REspEval.respeval.utils_json_output_process import load_json_robust
from REspEval.respeval.utils_TSP_flow_aggregate_plot import _handle_windows_path_length_limit
# -------------------------------
# Public API
# -------------------------------
def convincingness_specificity_directness_analysis(output, data_dir: Path, redo_eval=False, review='', response='',
                                        gen_type='') -> Dict[str, Any]:
    """
    """
    #### initialize scorer
    scorer = ConvSpecDirectScorer(data_dir=data_dir, 
                            openai_key='.keys/azure_key.txt',
                            openai_model='gpt-5')
    
    outfile = data_dir / f"@{gen_type}_conv_spec_direct.json"
    outfile = Path(_handle_windows_path_length_limit(outfile))
    
    if outfile.exists() and not redo_eval:
        print(f"conv_spec_direct: scores already exists, skipping evaluation.")
        with open(outfile, "r") as f:
            content = f.read()   # string
            #out = json.loads(content)  # dict
            out = load_json_robust(content)
    else:
        out = scorer.get_score(output, review, response)
        with open(outfile, 'w') as f:
                json.dump(out, f, indent=4)

   

    if isinstance(out, str):
       out = load_json_robust(out)
    
    return out['overall']
     

if __name__ == "__main__":
    print("This module is not intended to be run as a script.")
