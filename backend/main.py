from __future__ import annotations

import logging
import os
import time as clock
import uuid
from datetime import date, datetime, time, timedelta, timezone
from math import ceil
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from rag import (
    ConfigurationError,
    CorpusNotIndexedError,
    GroundingError,
    REFUSAL,
    RetrievalService,
)
from providers import ProviderError

logger = logging.getLogger(__name__)
app = FastAPI(title="SupportDesk-RAG API")
DAILY_REQUEST_LIMIT = int(os.getenv("DAILY_REQUEST_LIMIT", "10"))
_daily_request_counts: dict[str, tuple[date, int]] = {}
_daily_request_counts_lock = Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"
if DOCS_DIR.exists():
    app.mount("/docs", StaticFiles(directory=str(DOCS_DIR)), name="docs")

_service: RetrievalService | None = None


@app.middleware("http")
async def request_observability(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    started = clock.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request_failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
            },
        )
        raise
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round((clock.perf_counter() - started) * 1000, 2),
        },
    )
    return response


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)

    @field_validator("content")
    @classmethod
    def non_blank_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message content must not be blank")
        return value


class RetrievalRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    region: str | None = None
    category: str | None = None
    effective_date_from: str | None = None
    effective_date_to: str | None = None

    @field_validator("question")
    @classmethod
    def non_blank_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value

    @field_validator("effective_date_from", "effective_date_to")
    @classmethod
    def validate_date(cls, value: str | None) -> str | None:
        if value is None:
            return value
        from datetime import date

        try:
            return date.fromisoformat(value).isoformat()
        except ValueError as exc:
            raise ValueError("dates must use YYYY-MM-DD format") from exc

    @model_validator(mode="after")
    def validate_date_range(self) -> "RetrievalRequest":
        if (
            self.effective_date_from
            and self.effective_date_to
            and self.effective_date_from > self.effective_date_to
        ):
            raise ValueError("effective_date_from must not be after effective_date_to")
        return self

    def filters(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "region": self.region,
                "category": self.category,
                "effective_date_from": self.effective_date_from,
                "effective_date_to": self.effective_date_to,
            }.items()
            if value is not None
        }


class Citation(BaseModel):
    id: str
    document_id: str
    label: str
    pdf_filename: str
    pdf_url: str
    page: int
    section: str
    highlight_text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]


def get_service() -> RetrievalService:
    global _service
    if _service is None:
        try:
            _service = RetrievalService.from_environment()
        except (ConfigurationError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _service


def pdf_url_for(filename: str) -> str:
    return f"/docs/{quote(filename)}"


def retrieve_chunks(
    payload: RetrievalRequest,
    service: RetrievalService,
) -> list[dict[str, Any]]:
    try:
        return service.retrieve(payload.question, payload.filters())
    except CorpusNotIndexedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ProviderError as exc:
        logger.warning(
            "provider_failure",
            extra={
                "provider": exc.provider,
                "provider_status": exc.status_code,
                "retry_after": exc.retry_after,
                "stage": "retrieval",
            },
        )
        headers = {"Retry-After": exc.retry_after} if exc.retry_after else None
        raise HTTPException(
            status_code=502,
            detail="Upstream retrieval provider failed.",
            headers=headers,
        ) from exc
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(
            status_code=502, detail="Retrieval service failed."
        ) from exc


def consume_daily_request(ip_address: str, now: datetime | None = None) -> int | None:
    current_time = now or datetime.now(timezone.utc)
    current_day = current_time.date()

    with _daily_request_counts_lock:
        previous_day, count = _daily_request_counts.get(ip_address, (current_day, 0))
        if previous_day != current_day:
            count = 0

        if count >= DAILY_REQUEST_LIMIT:
            reset_at = datetime.combine(
                current_day + timedelta(days=1), time.min, tzinfo=timezone.utc
            )
            return max(1, ceil((reset_at - current_time).total_seconds()))

        _daily_request_counts[ip_address] = (current_day, count + 1)
        if len(_daily_request_counts) > 1024:
            for stale_ip, (stored_day, _) in tuple(_daily_request_counts.items()):
                if stored_day < current_day:
                    del _daily_request_counts[stale_ip]

    return None


def enforce_daily_ip_limit(request: Request) -> None:
    forwarded_for = request.headers.get("x-forwarded-for")
    ip_address = forwarded_for.split(",", 1)[0].strip() if forwarded_for else ""
    if not ip_address:
        ip_address = request.client.host if request.client else "unknown"
    retry_after = consume_daily_request(ip_address)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="Daily demo request limit reached. Try again after midnight UTC.",
            headers={"Retry-After": str(retry_after)},
        )


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
def ask_question(
    payload: RetrievalRequest,
    _limit: None = Depends(enforce_daily_ip_limit),
    service: RetrievalService = Depends(get_service),
) -> AskResponse:
    chunks = retrieve_chunks(payload, service)
    if not chunks:
        return AskResponse(question=payload.question, answer=REFUSAL, citations=[])

    try:
        answer, cited_sources = service.generate_answer(
            payload.question,
            chunks,
            history=[message.model_dump() for message in payload.history],
        )
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GroundingError as exc:
        logger.warning("Grounded answer rejected: %s", exc)
        raise HTTPException(
            status_code=502, detail="Grounded answer validation failed."
        ) from exc
    except ProviderError as exc:
        logger.warning(
            "provider_failure",
            extra={
                "provider": exc.provider,
                "provider_status": exc.status_code,
                "retry_after": exc.retry_after,
                "stage": "generation",
            },
        )
        headers = {"Retry-After": exc.retry_after} if exc.retry_after else None
        raise HTTPException(
            status_code=502,
            detail="Upstream generation provider failed.",
            headers=headers,
        ) from exc
    except Exception as exc:
        logger.exception("Answer generation failed")
        raise HTTPException(
            status_code=502, detail="Answer generation failed."
        ) from exc

    chunks_by_source: dict[tuple[str, str], dict[str, Any]] = {}
    for chunk in chunks:
        chunks_by_source.setdefault(
            (chunk["document_id"], chunk["section_title"]), chunk
        )
    citations = []
    for index, source in enumerate(cited_sources, start=1):
        chunk = chunks_by_source[source]
        filename = chunk["source_file"]
        page = chunk["page_start"]
        citations.append(
            Citation(
                id=str(index),
                document_id=chunk["document_id"],
                label=f"{filename} · p.{page}",
                pdf_filename=filename,
                pdf_url=pdf_url_for(filename),
                page=page,
                section=chunk["section_title"],
                highlight_text=chunk["content"],
            )
        )
    return AskResponse(question=payload.question, answer=answer, citations=citations)
