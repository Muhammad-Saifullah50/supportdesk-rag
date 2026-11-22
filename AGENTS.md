# AGENTS.md

## Project Overview

SupportDesk-RAG is an audit-grade, grounded retrieval assistant for enterprise corpora that may contain conflicting runbooks, dated policy manuals, and amended contracts. The system must prefer factual verification over fluent speculation.

The technical specification is `Technical Specification_ Grounded Retrieval Assistant.pdf`. Treat it as the behavioral source of truth. The implementation has a Next.js App Router frontend in `frontend/` and a FastAPI retrieval backend in `backend/`; the specification itself does not prescribe these technologies.

## Core Guarantees

- Every factual statement in a generated answer must have an exact inline citation.
- The assistant must use only evidence supplied in the retrieved `CONTEXT`.
- The assistant must not use outside knowledge, extrapolate, guess, or silently resolve conflicting evidence.
- Questions with insufficient evidence must receive exactly:

  ```text
  I do not have sufficient information in the provided corpus to answer this question.
  ```

- When sources conflict, explicitly describe the contradiction and cite every conflicting source.
- Regional and version differences must remain visible in the answer.
- If no evidence survives reranking, do not invoke the LLM; return the exact refusal string directly.

## Retrieval Pipeline

Implement and preserve this order:

1. Apply exact metadata filters before similarity evaluation. Supported filters include region, date range, and system/category.
2. Run hybrid retrieval using dense semantic search and BM25 lexical search.
3. Fuse the result lists with Reciprocal Rank Fusion (RRF).
4. Keep the top 15 RRF chunks.
5. Rerank those chunks with a cross-encoder.
6. Discard chunks with a cross-encoder score below `0.35`.
7. If chunks remain, pass only those chunks to the grounded generation prompt.
8. Generate an answer whose factual statements end with the required citation format.

Do not move metadata filtering after vector retrieval, change the top-15 boundary, lower the confidence threshold, or call the LLM when no chunks survive without an explicit specification update.

## Ingestion and Chunking

- Parse source documents as Markdown with header awareness.
- Produce chunks of approximately 300-500 tokens.
- Use 50-token overlap between adjacent chunks.
- Preserve complete provenance metadata on every stored chunk.
- Prefer stable, deterministic chunk IDs.

Every chunk must contain these exact fields:

```json
{
  "chunk_id": "doc_telecom_eu_v2_sec_4",
  "document_id": "runbook_eu_routing",
  "document_title": "EU Region Network Traffic Runbook",
  "version": "2.4.0",
  "effective_date": "2026-01-15",
  "region": "EU-WEST",
  "category": "network_runbook",
  "section_title": "BGP Failover Procedure",
  "content": "..."
}
```

Required fields are `chunk_id`, `document_id`, `document_title`, `version`, `effective_date`, `region`, `category`, `section_title`, and `content`.

## Technology Direction

The specification names these components:

- Qdrant or an equivalent vector store
- `text-embedding-3-small` for dense embeddings. The current user-approved implementation override is `nvidia/nemotron-3.5-embed-1b:free` via OpenRouter (2048 dimensions); see `decisions.md`. Do not revert this override without an explicit request.
- BM25 for sparse retrieval
- Reciprocal Rank Fusion for hybrid result fusion
- A cross-encoder reranker, such as `ms-marco-MiniLM-L-6-v2`
- A header-aware Markdown parser
- An LLM constrained by the grounded system prompt

Do not assume a language, web framework, database schema, deployment platform, endpoint design, or package manager from the specification. Inspect the repository before adding implementation-specific instructions or commands.

## Grounded System Prompt

The generation prompt must enforce all of the following:

- Use only factual snippets in `CONTEXT`.
- Do not use outside knowledge.
- Do not extrapolate beyond the evidence.
- End every factual statement with:

  ```text
  [Doc: <document_id>, Section: <section_title>]
  ```

- Require the exact refusal string when explicit evidence is missing.
- Require an explicit contradiction when supplied snippets conflict, with citations to both sources.

Citation validation should be deterministic where practical. A citation must identify a retrieved document and section, not merely a URL, filename, or generic source label.

## Testing and Evaluation

Maintain tests for at least these categories:

- Direct retrieval: an unambiguous policy question returns a factual answer with exact citations. Target citation precision: `100%`.
- Contradiction: conflicting regional or versioned runbooks are identified explicitly and both sources are cited. Target detection rate: `>=95%`.
- Out of corpus: an unindexed topic or version returns only the exact refusal string. Target zero-hallucination rate: `100%`.

Also test that:

- Metadata filters run before vector scoring.
- Exactly the top 15 RRF results are passed to reranking.
- Chunks scoring below `0.35` are removed.
- No surviving chunks means no LLM invocation.
- Every factual answer statement has a valid document and section citation.
- Contradictory evidence is not silently merged into one unsupported answer.

When adding or changing retrieval behavior, include focused unit tests and at least one end-to-end fixture with known metadata, ranking, scores, and expected citations.

## Safety and Data Handling

- Treat retrieved documents as the sole authority for answers.
- Preserve document, section, version, region, category, and effective-date provenance through ingestion, retrieval, reranking, and response formatting.
- Use exact metadata filtering to prevent irrelevant or unauthorized corpus context from entering retrieval.
- Never conceal conflicts between documents or versions.
- Do not add claims about authentication, authorization, encryption, tenant isolation, retention, or logging unless the implementation and requirements explicitly define them.
- Never commit credentials, API keys, tokens, local vector-store data, or other sensitive corpus material.

## Repository Conventions

The technical specification does not define source directories, environment variables, or development commands. Follow the implementation already present in the repository. Verified commands are:

- Frontend development: from `frontend/`, run `npm run dev`.
- Frontend build: from `frontend/`, run `npm run build`.
- Frontend tests: from `frontend/`, run `npm run test`.
- Backend tests: from `backend/`, run `.venv/bin/python -m unittest discover -s tests`.
- Set `frontend/BACKEND_URL` (default `http://127.0.0.1:8000`) to configure the Next.js proxy to the FastAPI service.

Before making changes:

- Inspect the existing tree and package/configuration files.
- Follow the repository's existing formatter, linter, test runner, and naming conventions.
- Add implementation-specific commands to this file only after verifying them in the repository.
- Keep changes minimal and avoid introducing compatibility layers without a concrete consumer or persisted-data requirement.

## Change Checklist

- Confirm the change preserves grounded-only generation.
- Confirm the exact refusal behavior remains unchanged.
- Confirm citations are present and point to retrieved provenance.
- Confirm contradiction handling remains explicit.
- Confirm metadata filtering still precedes similarity scoring.
- Add or update tests for changed pipeline stages.
- Run the repository's verified checks and report any unavailable checks explicitly.
