from __future__ import annotations

from collections import defaultdict
from typing import Any

import httpx

from providers import post_with_retries


COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"


class CohereReranker:
    """Small HTTP client for Cohere's hosted reranking API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        http_client: Any | None = None,
    ) -> None:
        self.http = http_client or httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
        self._owns_http = http_client is None
        self.model = model

    def predict(
        self,
        pairs: list[tuple[str, str]],
        **_: Any,
    ) -> list[float]:
        """Return scores in the same order as the query/document pairs."""
        if not pairs:
            return []

        grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for index, (query, document) in enumerate(pairs):
            grouped[query].append((index, document))

        scores = [0.0] * len(pairs)
        for query, indexed_documents in grouped.items():
            response = post_with_retries(
                self.http.post,
                provider="cohere",
                url=COHERE_RERANK_URL,
                json={
                    "model": self.model,
                    "query": query,
                    "documents": [document for _, document in indexed_documents],
                    "top_n": len(indexed_documents),
                },
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            if len(results) != len(indexed_documents):
                raise RuntimeError("Cohere returned incomplete rerank results.")
            for result in results:
                result_index = result.get("index")
                score = result.get("relevance_score")
                if (
                    not isinstance(result_index, int)
                    or not 0 <= result_index < len(indexed_documents)
                    or not isinstance(score, (int, float))
                ):
                    raise RuntimeError("Cohere returned malformed rerank results.")
                original_index = indexed_documents[result_index][0]
                scores[original_index] = float(score)

        return scores

    def close(self) -> None:
        if self._owns_http:
            self.http.close()
