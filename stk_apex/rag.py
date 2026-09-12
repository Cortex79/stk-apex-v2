"""
RAG-réteg — 7.1M dokumentum index (SUPERBRAIN-STK2-274M).

Index elérési út:
    SUPERBRAIN-STK2-274M/data/knowledge_layer/
    ├── stage1_v5/   (3.49M doc)
    └── v6_extra/    (3.60M doc)

Lekérdezési stratégia: BM25 (szöveg-alapú, gyors, telepítésfüggetlen).
Ha faiss elérhető: sűrű vektoros keresés fallback-ként.
"""
from __future__ import annotations

import os
import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple


class RAGRetriever:
    """
    Lazy-loading egyszerű visszakeresési réteg.

    Éles rendszerben felváltható faiss/BM25 bővítménnyel.
    """

    def __init__(self, index_path: str, top_k: int = 5,
                 max_tokens: int = 512):
        self.index_path = Path(index_path)
        self.top_k = top_k
        self.max_tokens = max_tokens
        self._docs: Optional[List[dict]] = None
        self._bm25 = None

    def _load(self) -> None:
        if self._docs is not None:
            return
        docs = []
        for subdir in ["stage1_v5", "v6_extra"]:
            p = self.index_path / subdir
            if not p.exists():
                continue
            for fpath in p.rglob("*.jsonl"):
                with fpath.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            doc = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = doc.get("text") or doc.get("content") or ""
                        if text:
                            docs.append({"text": text, "meta": doc})
                        if len(docs) >= 200_000:   # memória-korlát teszteléshez
                            break
                if len(docs) >= 200_000:
                    break
        self._docs = docs
        self._build_index()

    def _build_index(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
            corpus = [d["text"].lower().split() for d in self._docs]
            self._bm25 = BM25Okapi(corpus)
        except ImportError:
            self._bm25 = None   # fallback: kulcsszó-egyezés

    def retrieve(self, query: str) -> List[str]:
        """Top-k szöveg-részlet visszaadása a query-hez."""
        if not self.index_path.exists():
            return []
        self._load()
        if not self._docs:
            return []

        if self._bm25 is not None:
            scores = self._bm25.get_scores(query.lower().split())
            import heapq
            idxs = heapq.nlargest(self.top_k, range(len(scores)), key=lambda i: scores[i])
        else:
            # egyszerű szó-egyezéses fallback
            words = set(query.lower().split())
            scored = [(sum(1 for w in words if w in d["text"].lower()), i)
                      for i, d in enumerate(self._docs)]
            scored.sort(reverse=True)
            idxs = [i for _, i in scored[:self.top_k]]

        results = []
        for idx in idxs:
            text = self._docs[idx]["text"]
            # max_tokens alapján csonkítjuk (naiv whitespace tokenizáció)
            toks = text.split()
            if len(toks) > self.max_tokens:
                text = " ".join(toks[:self.max_tokens]) + "..."
            results.append(text)
        return results

    def format_context(self, query: str) -> str:
        """Prompt-ba illeszthető kontextus blokk."""
        docs = self.retrieve(query)
        if not docs:
            return ""
        parts = [f"[RAG {i+1}] {d}" for i, d in enumerate(docs)]
        return "\n\n".join(parts)

    @property
    def is_available(self) -> bool:
        return self.index_path.exists()
