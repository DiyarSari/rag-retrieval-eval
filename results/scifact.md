## scifact (5,183 docs, 300 test queries)

| System | nDCG@10 | MRR@10 | Recall@10 | Recall@100 | ms / query |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.664 | 0.633 | 0.783 | 0.886 | 0.1 |
| Dense (all-MiniLM-L6-v2) | 0.645 | 0.605 | 0.783 | 0.925 | 9.7 |
| Hybrid RRF (BM25 + Dense) | **0.694** | 0.657 | **0.825** | **0.958** | 10.0 |
| Hybrid RRF (BM25 + Dense) → rerank | 0.689 | **0.658** | 0.816 | **0.958** | 4327.4 |

_CPU only (12 threads), Python 3.10.0, single-query latency._
