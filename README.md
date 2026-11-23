# SupportDesk-RAG

SupportDesk-RAG is an audit-grade retrieval assistant for enterprise documents that may contain conflicting versions, dated policies, and amended procedures. It answers only from retrieved corpus evidence, preserves version and regional differences, and attaches provenance to every factual answer.

## Demo

- Frontend: https://supportdesk-frontewnd.vercel.app
- Backend health: https://supportdesk-backend-muhammadsaifullah50s-projects.vercel.app/api/health

The interface supports inline citations and opens the cited PDF at the relevant page.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the pipeline, constraints, and rejected alternatives.

## Quick Start

Requirements: Python 3.12, Node.js 20+, `uv`, and a Qdrant collection populated with the indexed corpus.

```bash
cp backend/.env.example backend/.env
make setup
make test
make run
```

The API runs at `http://127.0.0.1:8000`; the frontend runs at `http://localhost:3000`.

Required backend variables are documented in `backend/.env.example`. Never commit `backend/.env`, API keys, Qdrant data, or generated corpus manifests.

## Commands

```bash
make setup       # install backend and frontend dependencies
make run         # run the FastAPI backend
make test        # run backend and frontend tests
make eval        # run the deployed API evaluation harness
make eval-retrieval # run deterministic local retrieval baseline
make compile     # validate Python syntax
```

Run an evaluation against another deployment with:

```bash
python evals/run_live_eval.py \
  --base-url https://your-backend.example.com \
  --output evals/runs/$(date -u +%Y-%m-%dT%H%M%SZ).json
```

The evaluator records answerable fact recall, exact refusal rate, citation validity, expected-source coverage, latency, category metrics, and provider/API failures. A provider failure is not counted as an answer-quality failure.

## Guarantees

- Metadata filters run before similarity scoring.
- Dense and BM25 results are fused with RRF.
- Exactly the top 15 fused chunks are reranked.
- Cohere scores below `0.35` are discarded.
- No surviving evidence means no LLM call and the exact refusal string.
- Generated factual claims require citations to retrieved document sections.
- Conflicting evidence remains explicit rather than being silently merged.

## Known Limitations

- The hosted OpenRouter and Cohere trial tiers can rate-limit live evaluations.
- The demo request limit is intentionally small and is not a production tenancy or abuse-control system.
- Production retrieval depends on the Qdrant collection being populated with complete provenance payloads.
- Live evaluation scores vary with upstream model availability; committed reports include the exact commit and dataset used.
