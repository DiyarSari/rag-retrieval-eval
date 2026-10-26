"""Retrievers sharing one interface: ``search(query, k) -> [(doc_id, score), ...]``."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from .data import Document

Hit = tuple[str, float]

_TOKEN = re.compile(r"[a-z0-9]+")
# Small English stopword list; enough to keep BM25 from rewarding glue words.
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "this to was were which with not no than then there these those into such "
    "their they them we our can may also been being do does did".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 10) -> list[Hit]: ...


class BM25:
    """Okapi BM25 over an inverted index (Lucene-style IDF, Anserini defaults)."""

    name = "BM25"

    def __init__(self, docs: Sequence[Document], k1: float = 0.9, b: float = 0.4):
        self.k1, self.b = k1, b
        self.ids = [d.id for d in docs]
        tokenized = [tokenize(d.full_text) for d in docs]
        lengths = np.array([len(t) for t in tokenized], dtype=np.float32)
        self.norm = k1 * (1 - b + b * lengths / lengths.mean())

        postings: dict[str, tuple[list[int], list[int]]] = {}
        for i, tokens in enumerate(tokenized):
            for term, tf in Counter(tokens).items():
                docs_, tfs = postings.setdefault(term, ([], []))
                docs_.append(i)
                tfs.append(tf)
        n = len(docs)
        self.index = {
            term: (np.array(d), np.array(tf, dtype=np.float32), math.log(1 + (n - len(d) + 0.5) / (len(d) + 0.5)))
            for term, (d, tf) in postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.ids), dtype=np.float32)
        for term in tokenize(query):
            if term not in self.index:
                continue
            docs, tf, idf = self.index[term]
            scores[docs] += idf * tf * (self.k1 + 1) / (tf + self.norm[docs])
        return scores

    def search(self, query: str, k: int = 10) -> list[Hit]:
        # Documents sharing no term with the query are not matches.
        return _top_k(self.ids, self.scores(query), k, drop_zeros=True)


def _top_k(ids: list[str], scores: np.ndarray, k: int, drop_zeros: bool = False) -> list[Hit]:
    k = min(k, len(ids))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top], kind="stable")]
    return [(ids[i], float(scores[i])) for i in top if not (drop_zeros and scores[i] <= 0)]
