# rag_retriever.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

import torch
from transformers import AutoModel, AutoTokenizer


# --------------------------- Small utils ---------------------------

def _zscore(xs: List[float]) -> List[float]:
    if not xs:
        return []
    arr = np.asarray(xs, dtype=float)
    if np.all(arr == arr[0]):
        return [0.0] * len(xs)
    mu = float(arr.mean())
    std = float(arr.std()) + 1e-12
    return ((arr - mu) / std).tolist()

def _cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T

def _tokenize(paragraphs: List[str]) -> List[List[str]]:
    splitter = re.compile(r"\w+", re.UNICODE)
    return [splitter.findall(p.lower()) for p in paragraphs]

def _heuristic_boost(query: str, para_texts: List[str]) -> List[float]:
    """Light boosts when review/fact mentions likely anchors (table/figure/etc.)."""
    rt = query.lower()
    cues = {
        "fig": ["fig", "figure"],
        "table": ["table", "tbl"],
        "ablation": ["ablation"],
        "theorem": ["theorem", "thm"],
        "proof": ["proof"],
        "section": ["section", "sec."],
        "dataset": ["dataset"],
        "appendix": ["appendix", "supplement"],
    }
    active = {k: any(c in rt for c in vs) for k, vs in cues.items()}
    boosts = []
    for p in para_texts:
        pl = p.lower()
        b = 0.0
        if active["fig"] and ("fig" in pl or "figure" in pl): b += 0.15
        if active["table"] and ("table" in pl or "tbl" in pl): b += 0.15
        if active["ablation"] and "ablation" in pl: b += 0.10
        if active["theorem"] and ("theorem" in pl or "thm" in pl): b += 0.08
        if active["proof"] and "proof" in pl: b += 0.08
        if active["section"] and ("section" in pl or "sec." in pl): b += 0.05
        if active["dataset"] and "dataset" in pl: b += 0.05
        if active["appendix"] and ("appendix" in pl or "supplement" in pl): b += 0.03
        boosts.append(b)
    return boosts

def _rrf_fuse(rank_lists: List[List[int]], k: int, k_rrf: int = 60) -> List[int]:
    """Reciprocal Rank Fusion over index lists. Returns top-k fused indices."""
    scores: Dict[int, float] = {}
    for rl in rank_lists:
        for r, idx in enumerate(rl[:k_rrf]):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (60.0 + r + 1)
    return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])][:k]

def _hyde_stub(query: str) -> Optional[str]:
    """Optional HyDE hook. Return a synthetic paragraph to add as an extra query (or None)."""
    return None


# --------------------------- Embedding presets ---------------------------

EMBEDDING_PRESETS = {
    # Sentence-Transformers backend
    "bge-m3": {
        "hf_name": "BAAI/bge-m3",
        "backend": "st",
        "normalize": True,
        "query_instruction": "query: ",
        "doc_instruction": "passage: ",
        "trust_remote_code": False,
        "batch_size": 64,
        "max_length": 512,
    },
    # Science-tuned encoder (recommended for papers)
    "specter2": {
        "hf_name": "allenai/specter2_base",
        "backend": "st",
        "normalize": True,
        "query_instruction": None,
        "doc_instruction": None,
        "trust_remote_code": False,
        "batch_size": 64,
        "max_length": 512,
    },
    # Transformers path (mean pooling)
    "qwen3-embed-8b": {
        "hf_name": "Qwen/Qwen3-Embedding-8B",
        "backend": "hf",
        "normalize": True,
        "query_instruction": None,
        "doc_instruction": None,
        "trust_remote_code": True,
        "batch_size": 16,
        "max_length": 512,
    },
    # NV-Embed v2 via ST
    "nv-embed-v2": {
        "hf_name": "nvidia/NV-Embed-v2",
        "backend": "st",
        "normalize": True,
        "query_instruction": None,
        "doc_instruction": None,
        "trust_remote_code": True,
        "batch_size": 64,
        "max_length": 512,
    },
}

class EmbeddingEncoder:
    """Pluggable encoder supporting SentenceTransformers ('st') and HF ('hf') backends."""

    def __init__(self, preset: str):
        if preset not in EMBEDDING_PRESETS:
            raise ValueError(f"Unknown embedding preset: {preset}")
        self.cfg = EMBEDDING_PRESETS[preset]
        self.backend = self.cfg["backend"]
        self.normalize = bool(self.cfg.get("normalize", True))
        self.query_instruction = self.cfg.get("query_instruction")
        self.doc_instruction = self.cfg.get("doc_instruction")
        self.batch_size = int(self.cfg.get("batch_size", 32))
        self.max_length = int(self.cfg.get("max_length", 512))

        if self.backend == "st":
            self.model = SentenceTransformer(
                self.cfg["hf_name"],
                trust_remote_code=self.cfg.get("trust_remote_code", False),
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg["hf_name"], trust_remote_code=self.cfg.get("trust_remote_code", False)
            )
            self.model = AutoModel.from_pretrained(
                self.cfg["hf_name"], trust_remote_code=self.cfg.get("trust_remote_code", False)
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()

    def _apply_instruction(self, texts: List[str], is_query: bool) -> List[str]:
        ins = self.query_instruction if is_query else self.doc_instruction
        return [ins + t for t in texts] if ins else texts

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if self.backend == "st":
            texts = self._apply_instruction(texts, is_query)
            vecs = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            return vecs
        # HF mean pooling
        texts = self._apply_instruction(texts, is_query)
        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs)
                last_hidden = out.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                emb = (summed / counts).cpu().numpy()
            all_embs.append(emb)
        embs = np.vstack(all_embs)
        if self.normalize:
            embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        return embs


# --------------------------- RAG Retriever ---------------------------

@dataclass
class RAGRetriever:
    """
    RAG retriever for: given queries (fact, review, or fact+review gist) and a paper's paragraphs,
    return the most relevant paragraphs using a SOTA stack:
      - Hybrid first-stage: BM25 (sparse) + Dense (pluggable)
      - RRF fusion (across base rankers and across multiple queries)
      - Cross-encoder reranking (BAAI/bge-reranker-v2-m3)
      - Optional HyDE query augmentation (disabled by default)

    Typical use:
        retr = RAGRetriever(embedding_preset="specter2")
        retr.index(v1_doc_paragraphs)                 # list[str]
        hits = retr.retrieve_multi([fact, fact + ctx], top_k=12)
    """
    embedding_preset: str = "specter2"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    use_hyde: bool = False

    # First-stage depth (raise for long docs)
    rrf_kmax: int = 120          # depth per base ranker per query for RRF
    pre_k: int = 100             # how many after RRF to send to reranker

    # Final selection
    top_k_default: int = 12      # how many to return after reranking

    # internals
    _bm25: Optional[BM25Okapi] = None
    _paras: Optional[List[str]] = None
    _para_emb: Optional[np.ndarray] = None

    def __post_init__(self):
        self.encoder = EmbeddingEncoder(self.embedding_preset)
        self.reranker = CrossEncoder(self.reranker_model_name)
        self.top_k_default = int(self.top_k_default)

    # ---------------- index ----------------
    def index(self, paragraphs: List[str]):
        if not paragraphs:
            raise ValueError("Paragraph list is empty.")
        self._paras = paragraphs
        self._bm25 = BM25Okapi(_tokenize(paragraphs))
        self._para_emb = self.encoder.encode(paragraphs, is_query=False)

    # ---------------- base rankers ----------------
    def _dense_rank(self, query: str, topk: int) -> List[int]:
        assert self._para_emb is not None
        q = self.encoder.encode([query], is_query=True)[0]
        sims = (self._para_emb @ q)  # normalized if encoder.normalize=True
        return list(np.argsort(-sims)[:topk])

    def _bm25_rank(self, query: str, topk: int) -> List[int]:
        assert self._bm25 is not None
        toks = _tokenize([query])[0]
        scores = self._bm25.get_scores(toks)
        return list(np.argsort(-scores)[:topk])

    # ---------------- retrieval APIs ----------------
    def retrieve(self, query_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Single-query convenience wrapper (uses RRF over BM25+dense for one query)."""
        return self.retrieve_multi([query_text], top_k=top_k)

    def retrieve_multi(self, queries: List[str], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Multi-query retrieval:
          - For each query: get BM25 and Dense rank lists (depth rrf_kmax)
          - Fuse all lists via RRF to pre_k candidates
          - Rerank with cross-encoder (query=queries[0] by default)
          - Return top_k passages (adaptive if top score weak)
        """
        if self._bm25 is None or self._para_emb is None:
            raise RuntimeError("Call index(paragraphs) before retrieve/retrieve_multi.")

        qlist = list(queries)
        if self.use_hyde:
            hyde = _hyde_stub(queries[0])
            if hyde:
                qlist.append(hyde)

        # gather base ranks over all queries
        rank_lists: List[List[int]] = []
        for q in qlist:
            rank_lists.append(self._bm25_rank(q, self.rrf_kmax))
            rank_lists.append(self._dense_rank(q, self.rrf_kmax))

        # fuse and prepare candidates
        k_rerank = max(self.pre_k, top_k or self.top_k_default)
        cand_idx = _rrf_fuse(rank_lists, k=k_rerank, k_rrf=self.rrf_kmax)

        # rerank with cross-encoder (pair with the primary query)
        primary_q = qlist[0]
        pairs = [(primary_q, self._paras[i]) for i in cand_idx]
        scores = self.reranker.predict(pairs).tolist()
        order = np.argsort(-np.array(scores))

        pick_n = (top_k or self.top_k_default)
        # adaptive escalation: if top score weak, return more passages (up to 20)
        #if len(scores) and scores[order[0]] < 0.35:
        #    pick_n = min(max(pick_n, 20), len(order))

        out = []
        for r, i in enumerate(order[:pick_n], start=1):
            pid = int(cand_idx[i])
            out.append({
                "rank": r,
                "para_id": pid,
                "text": self._paras[pid],
                "rerank_score": float(scores[i]),
            })
        return out


# --------------------------- Example ---------------------------

if __name__ == "__main__":
    example_review = (
        "The evaluation lacks robustness analysis and does not compare strong baselines on dataset B. "
        "Please report ablations for loss terms and include Table 2 results with confidence intervals."
    )
    example_doc = [
        "We introduce a transformer-based architecture with multi-head attention.",
        "Figure 3 shows qualitative examples from dataset A.",
        "Ablation study: We remove each loss term (L_rec, L_kl) and report the impact on accuracy.",
        "Table 2: Comparison with baseline methods on dataset B, including 95% confidence intervals.",
        "We provide a robustness analysis by adding Gaussian noise and varying hyperparameters.",
        "Related work discusses prior methods on sequence modeling.",
    ]

    retr = RAGRetriever(embedding_preset="specter2", use_hyde=False)
    retr.index(example_doc)

    # Single-query (fact) or multi-query (fact + context)
    fact = "The paper includes Table 2 with baselines and 95% confidence intervals on dataset B."
    hits = retr.retrieve_multi([fact, f"{fact}. Context: {example_review}"], top_k=8)

    for h in hits:
        print(f"Rank {h['rank']} | para_id {h['para_id']} | score {h['rerank_score']:.3f}\n{h['text']}\n---")
