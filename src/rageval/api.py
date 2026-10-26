"""HTTP API over the benchmarked retrievers.

uvicorn rageval.api:app
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import data
from .answer import Answer, answer
from .retrievers import BM25, Dense, Hybrid, Retriever

DATASET = os.getenv("RAGEVAL_DATASET", "scifact")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ds = data.load(DATASET)
    docs = list(ds.corpus.values())
    bm25 = BM25(docs)
    dense = Dense(docs, cache_dir=data.DEFAULT_DATA_DIR / "cache" / DATASET)
    state["corpus"] = ds.corpus
    state["retrievers"] = {"bm25": bm25, "dense": dense, "hybrid": Hybrid([bm25, dense])}
    yield
    state.clear()


app = FastAPI(title="rag-retrieval-eval", lifespan=lifespan)


class SearchHit(BaseModel):
    id: str
    title: str
    score: float


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(5, ge=1, le=20)
    retriever: Literal["bm25", "dense", "hybrid"] = "hybrid"


class AskResponse(BaseModel):
    answer: str
    citations: list[dict]
    sources: list[SearchHit]
    refused: bool


def _search(q: str, k: int, retriever: str) -> list[SearchHit]:
    r: Retriever = state["retrievers"][retriever]
    corpus = state["corpus"]
    return [SearchHit(id=d, title=corpus[d].title, score=s) for d, s in r.search(q, k)]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "dataset": DATASET, "documents": len(state.get("corpus", {}))}


@app.get("/search", response_model=list[SearchHit])
def search(q: str, k: int = 10, retriever: Literal["bm25", "dense", "hybrid"] = "hybrid"):
    return _search(q, min(max(k, 1), 100), retriever)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    hits = _search(req.question, req.k, req.retriever)
    docs = [state["corpus"][h.id] for h in hits]
    try:
        result: Answer = answer(req.question, docs)
    except TypeError as e:
        # The SDK raises TypeError when no credential source is configured at all.
        if "authentication" not in str(e):
            raise
        raise HTTPException(503, "Set ANTHROPIC_API_KEY to enable /ask.") from None
    except anthropic.AuthenticationError:
        raise HTTPException(503, "Set ANTHROPIC_API_KEY to enable /ask.") from None
    except anthropic.RateLimitError:
        raise HTTPException(429, "Upstream rate limit, retry shortly.") from None
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"Upstream error ({e.status_code}).") from None
    except anthropic.APIConnectionError:
        raise HTTPException(502, "Could not reach the model API.") from None
    return AskResponse(
        answer=result.text,
        citations=[c.__dict__ for c in result.citations],
        sources=hits,
        refused=result.refused,
    )
