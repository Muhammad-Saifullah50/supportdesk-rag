from __future__ import annotations

import time
from typing import Any, Callable

import httpx


class ProviderError(RuntimeError):
    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retry_after = retry_after


def post_with_retries(
    post: Callable[..., Any],
    *,
    provider: str,
    url: str,
    json: dict[str, Any],
    max_retries: int = 2,
) -> Any:
    """Retry only transient provider failures and preserve useful diagnostics."""
    for attempt in range(max_retries + 1):
        try:
            response = post(url, json=json)
        except httpx.TimeoutException as exc:
            if attempt < max_retries:
                time.sleep(0.2 * (2**attempt))
                continue
            raise ProviderError(provider, "Provider request timed out.") from exc
        except httpx.RequestError as exc:
            if attempt < max_retries:
                time.sleep(0.2 * (2**attempt))
                continue
            raise ProviderError(provider, "Provider request failed.") from exc

        status_code = getattr(response, "status_code", None)
        if not isinstance(status_code, int):
            status_code = 200
        if status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
            time.sleep(0.2 * (2**attempt))
            continue
        if status_code >= 400:
            raise ProviderError(
                provider,
                f"{provider} returned HTTP {status_code}.",
                status_code=status_code,
                retry_after=getattr(response, "headers", {}).get("Retry-After"),
            )
        return response

    raise ProviderError(provider, "Provider request failed after retries.")
