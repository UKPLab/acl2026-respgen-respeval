# gfp_scorer.py
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from REspEval.respeval.rag_retriever import RAGRetriever
from REspEval.respeval.openai_lm import OpenAIModel

NLI_MODEL = 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
APPROACHES = {"RAG+GPT", "RAG+NLI"}

import logging, sys

logging.basicConfig(
    level=logging.INFO,                      # show INFO and above
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,                       # send to stdout (default is stderr)
    force=True                               # override existing handlers (Py>=3.8)
)

class AtomicFactScorer:
    """
    Modes:
      - 'RAG+GPT' : retrieval if source registered as RAG, else static; judge with GPT
      - 'RAG+NLI' : retrieval if source registered as RAG, else static; judge with NLI
    """

    def __init__(
        self,
        approach_name: str = "RAG+GPT",
        data_dir: str = "",
        openai_key: str = "api.key",
        openai_model: str = "gpt-5",
        retriever_embed: str = "specter2",
        device: Optional[str] = None,
        cache_dir: str = ".cache/respeval/cache_files",
    ):
        assert approach_name in APPROACHES, f"approach_name must be one of {APPROACHES}"
        self.approach_name = approach_name
        self.cache_dir = cache_dir
       
        self.data_dir = Path(data_dir)
        # LLM (only for GPT mode)
        self.openai_model = openai_model
        self.lm = OpenAIModel(
            openai_model,
            key_path=openai_key,
            cache_file=Path(self.cache_dir) / f"{openai_model}_atomic_fact_checks.db",
        ) if self.approach_name == "RAG+GPT" else None

        # NLI (only for NLI mode)
        self.device = device
        if self.approach_name == "RAG+NLI":
            self.nli_tok = AutoTokenizer.from_pretrained(NLI_MODEL)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
            if device:
                self.nli_model.to(device)
            self.nli_model.eval()

        # Knowledge sources
        self.sources: Dict[str, Dict[str, Any]] = {}
        self._retriever_embed = retriever_embed

      
    

    # ---------- Register sources ----------
    def register_rag_source(self, name: str, paragraphs: List[str]):
        """Register a long document for RAG retrieval."""
        assert name not in self.sources, f"{name} already registered"
        rag = RAGRetriever(embedding_preset=self._retriever_embed)
        rag.index(paragraphs)
        self.sources[name] = {"type": "RAG", "rag": rag, "passages": None}

    def register_static_source(self, name: str, passages: List[str]):
        """Register a short list of passages (no retrieval)."""
        assert name not in self.sources, f"{name} already registered"
        self.sources[name] = {"type": "STATIC", "rag": None, "passages": passages}

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
        review_text: str,
        #generation_texts: List[str],
        atomic_facts: List[Dict[str, Any]],
        knowledge_source: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        assert atomic_facts, "atomic_facts must be provided"
        assert knowledge_source in self.sources, f"{knowledge_source} not registered"
        src = self.sources[knowledge_source]

        decisions: List[Dict[str, Any]] = []
        input_words = output_words = 0
        t0 = time.time()

        # helpers
        def _ctx_from_passages(passages_text: List[str]) -> str:
            return "\n\n".join([f"Text: {p}" for p in passages_text]) if passages_text else "Text: "

        def _parse_label_json(ans: str) -> Optional[str]:
            low = ans.lower()
            if '"label"' in low:
                if "unsupported" in low: return "unsupported"
                if "contradicted" in low: return "contradicted"
                if "supported" in low: return "supported"       
            return None

        def _fallback_true_false(ans: str) -> str:
            low = ans.lower()
            if "true" in low and "false" not in low: return "supported"
            if "false" in low and "true" not in low: return "contradicted"
            return "unsupported"

        @torch.no_grad()
        def _nli_label(fact: str, passages_text: List[str]) -> str:
            if not passages_text:
                return "unsupported"
            best_ent, best_contra = -1e9, -1e9
            for txt in passages_text[:20]:
                pair = self.nli_tok(txt, fact, truncation=True, padding=True, return_tensors="pt")
                if self.device:
                    pair = {k: v.to(self.device) for k, v in pair.items()}
                logits = self.nli_model(**pair).logits[0].cpu().numpy()
                c, _, e = logits.tolist()
                best_ent = max(best_ent, e)
                best_contra = max(best_contra, c)
            if best_contra > best_ent + 1.0: return "contradicted"
            if best_ent > best_contra + 1.0: return "supported"
            return "unsupported"

        # per-fact loop
        atomic_facts_flattened = [fact for af in atomic_facts for fact in af['facts']]
        logging.info(f"atomic fact scorer: Scoring {len(atomic_facts_flattened)} atomic facts using {self.approach_name} with source {knowledge_source}")
        
        for af in atomic_facts:
            text = af.get("text", "")
            facts = af.get("facts", [])
            for fact in facts:
                fact = fact.strip()
                # passages
                if src["type"] == "RAG":
                    rag: RAGRetriever = src["rag"]
                    k_return = top_k or rag.top_k_default
                    #hits = rag.retrieve(review_text + " " + fact, top_k=k_return)
                    hits = rag.retrieve_multi([fact, review_text], top_k=k_return)
                    passages_text = [h["text"] for h in hits]
                else:
                    passages_text: List[str] = src["passages"] or []

                # judge
                if self.approach_name == "RAG+GPT":
                    ctx = _ctx_from_passages(passages_text)
                    system_prompt = (
                        "Determine whether the INPUT claim is supported by ANY of the following TEXT snippets.\n"
                        "Answer strictly as JSON: {\"label\": \"supported|contradicted|unsupported\", \"evidence\": <short quote or \"\">}.\n\n"
                    )
                    user_input = (
                        f"TEXT:{ctx}\n\nINPUT: {fact} \nOUTPUT:"
                    )
                    prompt = system_prompt + user_input
                    input_words += len(prompt.split())
                    out = self.lm.generate(system_prompt=system_prompt, user_input=user_input, max_output_length=300)
                    ans = out[0] if isinstance(out, tuple) else out
                    output_words += len(ans.split())
                    label = _parse_label_json(ans) or _fallback_true_false(ans)

                    decisions.append({
                        "atom": fact,
                        "is_supported": (label == "supported"),
                        "label": label,
                        "supported_passages": passages_text,
                        "eval_output_text": ans.strip(),
                    })
                else:  # RAG+NLI
                    label = _nli_label(fact, passages_text)
                    decisions.append({
                        "atom": fact,
                        "is_supported": (label == "supported"),
                        "label": label,
                        "supported_passages": passages_text,
                    })

        score = float(np.mean([d["is_supported"] for d in decisions])) if decisions else 0.0
        supported_p = sum(d["label"] == "supported" for d in decisions) / len(decisions) if decisions else 0.0
        contradicted_p = sum(d["label"] == "contradicted" for d in decisions) / len(decisions) if decisions else 0.0
        unsupported_p = sum(d["label"] == "unsupported" for d in decisions) / len(decisions) if decisions else 0.0


        total_time = round(time.time() - t0, 4)
        total_cost = self.cost_estimates(input_words, output_words, model=self.openai_model, task="atomic fact checking")

        return {
            "score": score,
            "supported_p": supported_p,
            "contradicted_p": contradicted_p,
            "unsupported_p": unsupported_p,
            "n_decisions": len(decisions),
            "decisions": decisions,
            "meta": {
                "score_approach": self.approach_name,
                "score_llm_model": self.openai_model if self.lm else None,
                "score_nli_model": NLI_MODEL if self.approach_name == "RAG+NLI" else None,
                "total_time": total_time,
                "total_input_words": input_words if self.approach_name == "RAG+GPT" else None,
                "total_output_words": output_words if self.approach_name == "RAG+GPT" else None,
                "total_cost": total_cost if self.approach_name == "RAG+GPT" else None,
                "source_type": src["type"],
                "top_k_used": top_k,
            }
        }
