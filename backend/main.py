from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException
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

logger = logging.getLogger(__name__)
app = FastAPI(title="SupportDesk-RAG API")

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


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    version: str
    effective_date: str
    region: str
    category: str
    section_title: str
    content: str
    source_file: str
    page_start: int
    page_end: int
    rrf_score: float
    rerank_score: float


class RetrievalResponse(BaseModel):
    question: str
    chunks: list[RetrievedChunk]
    refusal: str | None = None


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
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(
            status_code=502, detail="Retrieval service failed."
        ) from exc


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/retrieve", response_model=RetrievalResponse)
def retrieve_endpoint(
    payload: RetrievalRequest,
    service: RetrievalService = Depends(get_service),
) -> RetrievalResponse:
    chunks = retrieve_chunks(payload, service)
    return RetrievalResponse(
        question=payload.question,
        chunks=[RetrievedChunk.model_validate(chunk) for chunk in chunks],
        refusal=REFUSAL if not chunks else None,
    )


@app.post("/api/ask", response_model=AskResponse)
def ask_question(
    payload: RetrievalRequest,
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
