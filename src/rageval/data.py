"""Download and load BEIR-format datasets (corpus, queries, qrels)."""

from __future__ import annotations

import csv
import io
import json
import ssl
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import certifi

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
DEFAULT_DATA_DIR = Path("data")


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        return f"{self.title}\n{self.text}" if self.title else self.text


@dataclass
class Dataset:
    name: str
    corpus: dict[str, Document]
    queries: dict[str, str]
    # query id -> {doc id: graded relevance}
    qrels: dict[str, dict[str, int]]


def download(name: str, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Fetch a BEIR dataset once and return its directory."""
    target = data_dir / name
    if (target / "corpus.jsonl").exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} ...")
    # certifi's bundle, not the OS store: Windows only fetches the HARICA root this
    # host chains to on demand, which Python's ssl module never triggers.
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(BEIR_URL.format(name=name), timeout=120, context=ctx) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(data_dir)
    return target


def load(name: str, split: str = "test", data_dir: Path = DEFAULT_DATA_DIR) -> Dataset:
    root = download(name, data_dir)

    corpus: dict[str, Document] = {}
    with open(root / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[row["_id"]] = Document(row["_id"], row.get("title", ""), row["text"])

    qrels: dict[str, dict[str, int]] = {}
    with open(root / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for qid, did, score in reader:
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)

    queries: dict[str, str] = {}
    with open(root / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["_id"] in qrels:
                queries[row["_id"]] = row["text"]

    return Dataset(name, corpus, queries, qrels)
