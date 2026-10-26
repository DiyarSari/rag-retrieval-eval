"""Grounded answers: retrieved documents go to Claude as citable document blocks,
so every claim in the answer points at the exact passage it came from."""

from __future__ import annotations

from dataclasses import dataclass, field

import anthropic

from .data import Document

MODEL = "claude-opus-5-5"

SYSTEM = (
    "You answer questions using only the provided scientific abstracts. "
    "If the abstracts do not contain the answer, say so plainly instead of guessing. "
    "Keep the answer to a short paragraph."
)


@dataclass
class Citation:
    doc_id: str
    quote: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False


def build_messages(question: str, docs: list[Document]) -> list[dict]:
    blocks: list[dict] = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": d.full_text},
            "title": d.id,
            "citations": {"enabled": True},
        }
        for d in docs
    ]
    blocks.append({"type": "text", "text": question})
    return [{"role": "user", "content": blocks}]


def answer(question: str, docs: list[Document], client: anthropic.Anthropic | None = None) -> Answer:
    client = client or anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=build_messages(question, docs),
        output_config={"effort": "medium"},
        # On a safety decline, re-run the request on Anthropic's recommended fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        return Answer(text="The model declined to answer this question.", refused=True)

    parts: list[str] = []
    citations: list[Citation] = []
    for block in response.content:
        if block.type != "text":
            continue
        parts.append(block.text)
        for c in block.citations or []:
            doc_id = docs[c.document_index].id
            parts.append(f" [{doc_id}]")
            citations.append(Citation(doc_id, c.cited_text.strip()))
    return Answer(text="".join(parts).strip(), citations=citations)
