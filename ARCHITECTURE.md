# Architecture and Decisions

## Problem

Support teams need answers over policy and runbook corpora where documents can conflict by version, region, or effective date. Fluent answers are not useful if an operator cannot verify the source or if an archived rule is presented as current.

## Request Flow

```text
Browser
  -> Next.js proxy
  -> FastAPI /api/ask
  -> exact metadata filtering
  -> dense Qdrant retrieval + BM25 lexical retrieval
  -> Reciprocal Rank Fusion
  -> top 15 chunks
  -> hosted Cohere cross-encoder reranking
  -> scores below 0.35 discarded
  -> grounded OpenRouter JSON generation
  -> deterministic citation validation
  -> cited answer and PDF provenance
```

## Decisions

### Hybrid retrieval

Dense retrieval handles paraphrases while BM25 preserves exact identifiers, error codes, plan names, and numeric policy terms. RRF combines their ranks without assuming the score scales are comparable.

### Filter before retrieval

Region, category, and effective-date filters are applied before semantic scoring. Filtering later would allow irrelevant or unauthorized evidence to influence ranking.

### Top-15 boundary and reranking

RRF produces a bounded candidate set of 15 chunks. Cohere's hosted reranker replaces the local PyTorch cross-encoder because Vercel's Python function limit cannot package the ML runtime. The threshold remains `0.35` and is tested as a pipeline invariant.

### Grounded generation

The LLM receives only surviving retrieved chunks and a strict JSON schema. The application validates every cited document and section against retrieved provenance before returning an answer.

### Refusal over speculation

When no evidence survives, the API returns the exact refusal string without invoking the LLM. This makes the out-of-corpus behavior deterministic and avoids confident unsupported answers.

### Serverless corpus loading

Local `chunks.jsonl` is preferred for development. In a serverless deployment, chunk payloads are loaded from Qdrant so generated local manifests do not need to be committed. PDFs are served from `frontend/public/docs` because frontend static assets are more reliable than Python-function filesystem packaging.

## Rejected Alternatives

- Local `sentence-transformers`: rejected because PyTorch and CUDA dependencies exceeded Vercel's 500 MB function limit.
- Qdrant-only reranking: rejected because vector similarity does not provide query-document cross-encoder relevance scoring.
- LLM-only retrieval: rejected because it cannot provide deterministic metadata filtering, ranking boundaries, or auditable provenance.
- Silent fallback on provider errors: rejected because it would make evaluation and incident diagnosis misleading.

## Scaling Considerations

At larger corpus sizes, chunk payload loading should move to a warm cache or a dedicated retrieval service, and BM25 should use a persistent sparse index. At higher traffic, the in-memory demo limiter should be replaced by a shared rate-limit store and provider budgets should be tracked per tenant.
