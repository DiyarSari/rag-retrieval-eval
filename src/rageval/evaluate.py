"""Benchmark every retriever on a BEIR dataset and write a results table.

python -m rageval.evaluate --dataset scifact
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

from . import data
from .metrics import evaluate
from .retrievers import BM25, Dense, Hybrid, Reranked, Retriever

DEPTH = 100  # every system returns 100 hits so Recall@100 is comparable


def build(ds: data.Dataset, cache_dir: Path, rerank: bool) -> list[Retriever]:
    docs = list(ds.corpus.values())
    bm25 = BM25(docs)
    dense = Dense(docs, cache_dir=cache_dir)
    hybrid = Hybrid([bm25, dense], depth=DEPTH)
    systems: list[Retriever] = [bm25, dense, hybrid]
    if rerank:
        systems.append(Reranked(hybrid, docs, depth=DEPTH))
    return systems


def run_system(system: Retriever, queries: dict[str, str]) -> tuple[dict[str, list[str]], float]:
    run, start = {}, time.perf_counter()
    for qid, text in queries.items():
        run[qid] = [doc for doc, _ in system.search(text, DEPTH)]
    ms_per_query = (time.perf_counter() - start) * 1000 / len(queries)
    return run, ms_per_query


def to_markdown(ds: data.Dataset, rows: list[dict]) -> str:
    metrics = ["nDCG@10", "MRR@10", "Recall@10", "Recall@100"]
    best = {m: max(r[m] for r in rows) for m in metrics}
    lines = [
        f"## {ds.name} ({len(ds.corpus):,} docs, {len(ds.queries)} test queries)",
        "",
        "| System | " + " | ".join(metrics) + " | ms / query |",
        "|---|" + "---:|" * (len(metrics) + 1),
    ]
    for r in rows:
        cells = [f"**{r[m]:.3f}**" if r[m] == best[m] else f"{r[m]:.3f}" for m in metrics]
        lines.append(f"| {r['system']} | " + " | ".join(cells) + f" | {r['ms_per_query']:.1f} |")
    lines += ["", f"_CPU only ({os.cpu_count()} threads), Python {platform.python_version()}, single-query latency._"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--data-dir", type=Path, default=data.DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--no-rerank", action="store_true", help="skip the (slow) cross-encoder stage")
    args = parser.parse_args()

    ds = data.load(args.dataset, data_dir=args.data_dir)
    systems = build(ds, args.data_dir / "cache" / args.dataset, rerank=not args.no_rerank)

    rows = []
    for system in systems:
        run, ms = run_system(system, ds.queries)
        row = {"system": system.name, **evaluate(run, ds.qrels), "ms_per_query": ms}
        rows.append(row)
        print(f"{system.name:<40} nDCG@10={row['nDCG@10']:.3f}  R@100={row['Recall@100']:.3f}  {ms:.1f} ms/q")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{ds.name}.json").write_text(json.dumps(rows, indent=2))
    (args.out / f"{ds.name}.md").write_text(to_markdown(ds, rows), encoding="utf-8")
    print(f"\nWrote {args.out / ds.name}.md")


if __name__ == "__main__":
    main()
