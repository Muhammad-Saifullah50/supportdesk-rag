# Architecture Decisions

## 1. Header-aware chunking

Documents are split around headings so logical clauses remain together. Chunks target approximately 300-500 tokens with a 50-token overlap. This preserves context without allowing unrelated sections to dilute the representation.

## 2. Metadata filtering before search

Region, category, and effective-date filters are applied before similarity evaluation. This prevents unrelated documents from influencing ranking and keeps the evidence set auditable.

## 3. Hybrid search

Dense embeddings handle paraphrased questions, while BM25 handles exact strings such as codes, numbers, and plan names. Combining both improves coverage across the sample corpus.

## 4. Reciprocal Rank Fusion

Dense and BM25 retrieval scores are not directly comparable. Reciprocal Rank Fusion combines their rankings without inventing a shared score scale.

## 5. Cross-encoder reranking

Initial retrieval encodes the question and document separately. A cross-encoder evaluates the question and chunk together, producing a stricter relevance score. It is applied only to the top 15 RRF results to control latency.

## 6. Score threshold and refusal circuit

The reranker threshold is `0.35`. If no chunk survives, the API returns the exact refusal message without invoking the language model. This makes out-of-corpus behavior deterministic.

## 7. Embedding provider and model

The implementation uses OpenRouter's `nvidia/nemotron-3.5-embed-1b:free` for document and query embeddings, returning 2048-dimensional vectors. Changing the model requires re-embedding the corpus and using a compatible Qdrant collection.
