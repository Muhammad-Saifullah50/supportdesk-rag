from __future__ import annotations

from typing import Any

import httpx

from providers import post_with_retries

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    def __init__(self, api_key: str, http_client: Any | None = None):
        self.http = http_client or httpx.Client(
            base_url=OPENROUTER_BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )
        self._owns_http = http_client is None
        self.last_usage: dict[str, Any] | None = None

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        response = post_with_retries(
            self.http.post,
            provider="openrouter",
            url="/embeddings",
            json={"model": model, "input": texts, "encoding_format": "float"},
        )
        response.raise_for_status()
        items = response.json().get("data", [])
        vectors: list[list[float] | None] = [None] * len(texts)
        for item in items:
            index = item.get("index")
            vector = item.get("embedding")
            if (
                not isinstance(index, int)
                or not 0 <= index < len(vectors)
                or not isinstance(vector, list)
            ):
                raise RuntimeError(
                    "OpenRouter returned a malformed embedding response."
                )
            vectors[index] = vector
        if any(vector is None for vector in vectors):
            raise RuntimeError(
                "OpenRouter did not return an embedding for every input."
            )
        return [vector for vector in vectors if vector is not None]

    def complete_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        response_schema: dict[str, Any],
    ) -> str:
        response = post_with_retries(
            self.http.post,
            provider="openrouter",
            url="/chat/completions",
            json={
                "model": model,
                "temperature": 0,
                "max_tokens": max_tokens,
                "provider": {"require_parameters": True},
                "reasoning": {"enabled": False},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": response_schema,
                },
                "messages": messages,
            },
        )
        response.raise_for_status()
        response_body = response.json()
        self.last_usage = response_body.get("usage")
        choices = response_body.get("choices", [])
        if not choices:
            raise RuntimeError("OpenRouter returned no completion choices.")
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                part["text"]
                for part in content
                if isinstance(part, dict)
                and part.get("type") in {"text", "output_text"}
                and isinstance(part.get("text"), str)
            )
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                "OpenRouter returned no text content "
                f"(finish_reason={choice.get('finish_reason')!r}, "
                f"native_finish_reason={choice.get('native_finish_reason')!r}, "
                f"message_fields={sorted(message) if isinstance(message, dict) else []}, "
                f"reasoning_present={bool(message.get('reasoning')) if isinstance(message, dict) else False}, "
                f"refusal_present={bool(message.get('refusal')) if isinstance(message, dict) else False})."
            )
        return content

    def close(self) -> None:
        if self._owns_http:
            self.http.close()
