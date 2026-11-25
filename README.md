# SupportDesk-RAG

SupportDesk-RAG is a portfolio demonstration of grounded document question answering. It answers questions over a small sample corpus, preserves version and regional differences, and attaches provenance to every factual answer.

## Demo

- Frontend: https://supportdesk-frontewnd.vercel.app
- Backend health: https://supportdesk-backend-sigma.vercel.app/api/health

The interface supports inline citations and opens the cited PDF at the relevant page. The deployed app uses fictional sample documents and is not intended for confidential or production data.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the pipeline, constraints, and rejected alternatives.

## Quick Start

Requirements: Python 3.12, Node.js 20+, `uv`, and a Qdrant collection populated with the indexed sample corpus.

```bash
cp backend/.env.example backend/.env
make setup
make test
make run
```

The API runs at `http://127.0.0.1:8000`; the frontend runs at `http://localhost:3000`.

Required backend variables are documented in `backend/.env.example`. Set `ALLOWED_ORIGINS` to a comma-separated list when using a different frontend URL. Never commit `backend/.env`, API keys, Qdrant data, or generated corpus manifests.

## Commands

```bash
make setup       # install backend and frontend dependencies
make run         # run the FastAPI backend
make test        # run backend and frontend tests
make eval        # run the deployed API evaluation harness
make eval-retrieval # run deterministic local retrieval baseline
make compile     # validate Python syntax
```

Run the evaluation against the deployed portfolio API with:

```bash
python evals/run_live_eval.py \
  --base-url https://supportdesk-backend-sigma.vercel.app \
  --output evals/runs/$(date -u +%Y-%m-%dT%H%M%SZ).json
```

The evaluator records answerable fact recall, exact refusal rate, citation validity, expected-source coverage, latency, category metrics, and provider/API failures. A provider failure is not counted as an answer-quality failure. The default smoke set contains 20 representative cases; the original 86-case set is preserved as `evals/eval_dataset.full.json`.

## Evaluation Results

Measured on the canonical 20-question smoke set (18 answerable, 2 out-of-corpus) against the deployed API. Report: [`evals/runs/2026-11-25T050023Z-merged-20.json`](evals/runs/2026-11-25T050023Z-merged-20.json), commit `3f9cf9b`.

| Metric | Result | What it means |
| --- | --- | --- |
| **Fully correct answers** | **14 / 20 (70%)** | Every gold key fact present, or the exact refusal for out-of-corpus questions |
| Answerable questions fully correct | 12 / 18 (67%) | All required facts stated with citations |
| Average key-fact recall | 75.5% | Share of gold facts that appear in the answers |
| Out-of-corpus refusal accuracy | 2 / 2 (100%) | Unanswerable questions return the exact refusal string, never a guess |
| Citation precision | 20 / 20 (100%) | Every factual statement carries document + section + PDF provenance |
| Contradiction/version questions correct | 2 / 3 (67%) | Conflicting or dated sources are surfaced, not silently merged |
| Section Recall@15 (BM25 baseline) | 100% | Every expected source section is retrieved in the top 15 |
| Document Recall@15 (BM25 baseline) | 100% | Every expected source document is retrieved |
| Retrieval MRR (BM25 baseline) | 0.62 | Rank position of the first correct section |
| Answer latency (p50 / p95) | 10.3s / 28.0s | End-to-end retrieval, reranking, and generation |
| Infrastructure errors in final run | 0 / 20 | Provider failures are excluded from answer-quality scores |

Known weaknesses: 2 of 18 answerable questions were refused despite evidence in the corpus (q046, q059), 4 answers stated the right conclusion with missing key facts (q034, q035, q067, q076), and 2 more were fact-complete but cited only part of the expected sources (q057, q058). The gap is answer completeness, not hallucination — no unsupported factual claim was observed.

## Guarantees

- Metadata filters run before similarity scoring.
- Dense and BM25 results are fused with RRF.
- Exactly the top 15 fused chunks are reranked.
- Cohere scores below `0.35` are discarded.
- No surviving evidence means no LLM call and the exact refusal string.
- Generated factual claims require citations to retrieved document sections.
- Conflicting evidence remains explicit rather than being silently merged.

## Portfolio Scope and Limitations

- The hosted OpenRouter and Cohere trial tiers can rate-limit live evaluations.
- The demo request limit is intentionally small and is not a production tenancy or abuse-control system.
- Retrieval depends on the Qdrant collection being populated with complete provenance payloads.
- The corpus contains fictional sample documents. Do not upload confidential company material.
- Live evaluation scores vary with upstream model availability; reports record the dataset and commit used.
