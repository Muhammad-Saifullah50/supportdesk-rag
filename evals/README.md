# Evaluation Workflow

`eval_dataset.json` is the canonical 20-question smoke set. The original 86-question set is preserved in `eval_dataset.full.json`. `build_evals.py` validates the smoke set and regenerates the line-oriented `eval_dataset.jsonl` form.

## Retrieval Baseline

Run the deterministic local BM25 baseline:

```bash
make eval-retrieval
```

This reports section Recall@15, document Recall@15, MRR, stale-document top-1 rate, and category breakdowns. It is a retrieval baseline, not the full dense/RRF/Cohere pipeline.

## Live Answer Evaluation

Run the deployed API evaluator:

```bash
EVAL_BASE_URL=https://supportdesk-backend-sigma.vercel.app make eval
```

The report separates answer-quality metrics from infrastructure failures. OpenRouter or Cohere `429`/`502` responses are recorded under `error_classes` and excluded from answer-quality denominators.

## Metrics

- `fact_recall`: fraction of gold key facts present in an answer.
- `exact_refusal_rate`: fraction of unanswerable questions returning the exact refusal string.
- `citation_valid_rate`: every returned citation contains document, section, and PDF provenance.
- `expected_document_coverage`: expected source documents represented in returned citations.
- `section_recall@15`: expected source section appears in the retrieval top-k.
- `mrr`: reciprocal rank of the first expected source section.
- `stale_top1_rate`: archived distractor is ranked first.

Live reports must record the commit, dataset, deployment URL, model configuration, and provider failures. Do not present a run dominated by provider errors as a model-quality result.
