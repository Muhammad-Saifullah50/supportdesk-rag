# Retrieval Pipeline

This portfolio project uses a grounded retrieval pipeline. The order is deliberate because each stage reduces the chance of irrelevant or unsupported evidence reaching answer generation.

## 1. Metadata pre-filtering

Region, category, and effective-date filters are applied before similarity scoring. For example, a regional question should not be ranked against documents from unrelated regions.

## 2. Hybrid retrieval

Two searches run over the filtered corpus:

- Dense search finds chunks with similar meaning using embeddings.
- BM25 lexical search preserves exact matches for codes, numbers, plan names, and error messages.

Reciprocal Rank Fusion combines the two ranked lists without requiring their raw scores to use the same scale.

## 3. Cross-encoder reranking

The pipeline keeps exactly the top 15 fused chunks. A cross-encoder compares the question and each chunk together, producing a stricter relevance score.

Chunks scoring below `0.35` are discarded.

## 4. Grounded generation

Only surviving chunks are passed to the language model. The answer validator checks that citations refer to retrieved document sections.

## 5. Refusal guardrail

If every chunk falls below the reranking threshold, the language model is not called. The API returns the exact refusal message instead of guessing.
