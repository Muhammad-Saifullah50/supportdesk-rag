from __future__ import annotations

import argparse # parses command line arguments when runnign the script
import json # reads and validates JSON
import math # provides mathematical methods
import re # matches numbered section headings
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

import tiktoken # encodes and decodes text into token IDs using OpenAI's tokenizer

from dotenv import load_dotenv

from openrouter import OpenRouterClient
from rag import (
    BACKEND_DIR,
    DEFAULT_COLLECTION,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
)

ROOT_DIR = BACKEND_DIR.parent
DOCS_DIR = ROOT_DIR / "docs"
MANIFEST_PATH = BACKEND_DIR / "ingestion_manifest.json"
OUTPUT_PATH = BACKEND_DIR / "data" / "chunks.jsonl"
COLLECTION_NAME = DEFAULT_COLLECTION
CHUNK_TARGET_TOKENS = 400
CHUNK_MIN_TOKENS = 300
CHUNK_MAX_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
EMBEDDING_BATCH_SIZE = 64
SECTION_NUMBER = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+[A-Z][^.!?]{1,130}$")
NON_WORD = re.compile(r"[^a-zA-Z0-9]+")


@dataclass
class Section:
    title: str
    blocks: list[tuple[str, int]] = field(default_factory=list)


def is_heading(text: str, font_size: float, median_font_size: float) -> bool:
    normalized = " ".join(text.split())
    if not normalized or len(normalized) > 160 or len(normalized.split()) > 20:
        return False
    if SECTION_NUMBER.match(normalized):
        return True
    letters = [character for character in normalized if character.isalpha()]
    if len(letters) >= 4 and normalized.upper() == normalized:
        return True
    return (
        font_size >= median_font_size * 1.22
        and len(normalized.split()) <= 16
        and not normalized.endswith((".", ";", ","))
    )


def extract_pdf_sections(pdf_path: Path, fallback_title: str) -> list[Section]:
    import fitz # extracts text, fonty sizes and layout blocks from  PDFs

    sections = [Section(title=fallback_title)]
    with fitz.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf, start=1):
            page_data = page.get_text("dict", sort=True)
            lines: list[tuple[str, float]] = []
            sizes: list[float] = []
            for block in page_data.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = "".join(span.get("text", "") for span in spans).strip()
                    if not text:
                        continue
                    size = max(
                        (float(span.get("size", 0)) for span in spans), default=0.0
                    )
                    lines.append((text, size))
                    sizes.extend(
                        float(span.get("size", 0))
                        for span in spans
                        if span.get("text", "").strip()
                    )
            median_size = statistics.median(sizes) if sizes else 0.0
            pending: list[str] = []

            def flush_pending() -> None:
                if pending:
                    sections[-1].blocks.append((" ".join(pending), page_index))
                    pending.clear()

            for text, size in lines:
                if is_heading(text, size, median_size):
                    flush_pending()
                    sections.append(Section(title=text))
                else:
                    pending.append(text)
            flush_pending()

    sections = [section for section in sections if section.blocks]
    if not sections:
        raise ValueError(
            f"No extractable text found in {pdf_path.name}; OCR is not performed."
        )
    return sections


def chunk_sections(
    sections: list[Section],
    document: dict[str, str],
    source_file: str,
) -> list[dict[str, Any]]:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    chunks: list[dict[str, Any]] = []

    for section in sections:
        tokens: list[int] = []
        token_pages: list[int] = []
        for block_text, page_number in section.blocks:
            if tokens:
                separator = tokenizer.encode("\n\n", disallowed_special=())
                tokens.extend(separator)
                token_pages.extend([page_number] * len(separator))
            block_tokens = tokenizer.encode(block_text, disallowed_special=())
            tokens.extend(block_tokens)
            token_pages.extend([page_number] * len(block_tokens))

        if not tokens:
            continue

        section_slug = (
            NON_WORD.sub("_", section.title.casefold()).strip("_")[:48] or "section"
        )
        minimum_chunks = (
            1
            if len(tokens) <= CHUNK_MAX_TOKENS
            else math.ceil(
                (len(tokens) - CHUNK_OVERLAP_TOKENS)
                / (CHUNK_MAX_TOKENS - CHUNK_OVERLAP_TOKENS)
            )
        )
        maximum_chunks = max(
            minimum_chunks,
            (len(tokens) - CHUNK_OVERLAP_TOKENS)
            // (CHUNK_MIN_TOKENS - CHUNK_OVERLAP_TOKENS),
        )
        chunk_count = min(
            range(minimum_chunks, maximum_chunks + 1),
            key=lambda count: abs(
                (len(tokens) + CHUNK_OVERLAP_TOKENS * (count - 1)) / count
                - CHUNK_TARGET_TOKENS
            ),
        )
        chunk_size = min(
            CHUNK_MAX_TOKENS,
            math.ceil(
                (len(tokens) + CHUNK_OVERLAP_TOKENS * (chunk_count - 1)) / chunk_count
            ),
        )
        stride = chunk_size - CHUNK_OVERLAP_TOKENS
        starts = (index * stride for index in range(chunk_count))
        for start in starts:
            end = min(start + chunk_size, len(tokens))
            content = tokenizer.decode(tokens[start:end]).strip()
            if not content:
                continue
            chunk_id = (
                f"{document['document_id']}_v{document['version']}_sec_"
                f"{section_slug}_{len(chunks):04d}"
            )
            pages = token_pages[start:end]
            effective_date = date.fromisoformat(document["effective_date"])
            effective_date_ts = datetime.combine(
                effective_date, time.min, tzinfo=timezone.utc
            ).isoformat()
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document["document_id"],
                    "document_title": document["document_title"],
                    "version": document["version"],
                    "effective_date": document["effective_date"],
                    "effective_date_ts": effective_date_ts,
                    "region": document["region"],
                    "category": document["category"],
                    "section_title": section.title,
                    "content": content,
                    "source_file": source_file,
                    "page_start": min(pages),
                    "page_end": max(pages),
                }
            )
            if end == len(tokens):
                break
    return chunks


def load_manifest(manifest_path: Path, docs_dir: Path) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        raise ValueError(
            f"Metadata manifest not found: {manifest_path}. "
            "Copy ingestion_manifest.example.json and fill in every required field."
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("The manifest must contain a non-empty 'documents' array.")

    required = (
        "source_file",
        "document_id",
        "document_title",
        "version",
        "effective_date",
        "region",
        "category",
    )
    filenames: set[str] = set()
    document_versions: set[tuple[str, str]] = set()
    normalized: list[dict[str, str]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entry {index} must be an object.")
        missing = [
            name
            for name in required
            if not isinstance(entry.get(name), str) or not entry[name].strip()
        ]
        if missing:
            raise ValueError(
                f"Manifest entry {index} is missing: {', '.join(missing)}."
            )
        source_file = entry["source_file"].strip()
        if Path(source_file).name != source_file:
            raise ValueError(
                f"source_file must be a filename directly under {docs_dir}: {source_file}"
            )
        if source_file in filenames:
            raise ValueError(f"Duplicate source_file in manifest: {source_file}")
        filenames.add(source_file)
        try:
            date.fromisoformat(entry["effective_date"])
        except ValueError as exc:
            raise ValueError(
                f"Invalid effective_date for {source_file}; expected YYYY-MM-DD."
            ) from exc
        identity = (entry["document_id"].strip(), entry["version"].strip())
        if identity in document_versions:
            raise ValueError(f"Duplicate document_id/version in manifest: {identity}")
        document_versions.add(identity)
        if not (docs_dir / source_file).is_file():
            raise ValueError(
                f"Manifest source does not exist: {docs_dir / source_file}"
            )
        normalized.append({key: entry[key].strip() for key in required})

    unlisted_pdfs = {
        path.name for path in docs_dir.glob("*.pdf") if path.name not in filenames
    }
    if unlisted_pdfs:
        raise ValueError(
            "Every PDF must have explicit metadata. Missing manifest entries for: "
            + ", ".join(sorted(unlisted_pdfs))
        )
    return normalized


def embed_chunks(
    client: OpenRouterClient, chunks: list[dict[str, Any]]
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
        batch_vectors = client.embed(
            [
                f"{chunk['document_title']}\n{chunk['section_title']}\n{chunk['content']}"
                for chunk in batch
            ],
            model=EMBEDDING_MODEL,
        )
        if len(batch_vectors) != len(batch):
            raise RuntimeError("OpenRouter returned the wrong number of embeddings.")
        vectors.extend(batch_vectors)
    if any(len(vector) != EMBEDDING_DIMENSIONS for vector in vectors):
        raise RuntimeError(
            "Embedding response was missing vectors or had unexpected dimensions."
        )
    return vectors


def ensure_collection(client: Any, collection_name: str) -> None:
    from qdrant_client import models

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=EMBEDDING_DIMENSIONS,
                    distance=models.Distance.COSINE,
                )
            },
        )
    else:
        collection = client.get_collection(collection_name)
        vectors = collection.config.params.vectors
        dense = vectors.get("dense") if isinstance(vectors, dict) else None
        if (
            dense is None
            or dense.size != EMBEDDING_DIMENSIONS
            or dense.distance != models.Distance.COSINE
        ):
            raise RuntimeError(
                f"Collection {collection_name} exists with an incompatible dense vector schema."
            )

    indexes = {
        "region": models.PayloadSchemaType.KEYWORD,
        "category": models.PayloadSchemaType.KEYWORD,
        "document_id": models.PayloadSchemaType.KEYWORD,
        "version": models.PayloadSchemaType.KEYWORD,
        "effective_date_ts": models.PayloadSchemaType.DATETIME,
    }
    for field_name, field_schema in indexes.items():
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
            wait=True,
        )


def qdrant_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"supportdesk-rag:{chunk_id}"))


def remove_stale_points(
    client: Any,
    collection_name: str,
    chunks: list[dict[str, Any]],
) -> None:
    from qdrant_client import models

    by_document_version: dict[tuple[str, str], set[str]] = {}
    for chunk in chunks:
        by_document_version.setdefault(
            (chunk["document_id"], chunk["version"]), set()
        ).add(qdrant_point_id(chunk["chunk_id"]))

    for (document_id, version), current_ids in by_document_version.items():
        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id", match=models.MatchValue(value=document_id)
                ),
                models.FieldCondition(
                    key="version", match=models.MatchValue(value=version)
                ),
            ]
        )
        offset = None
        stale_ids: list[str | int] = []
        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            stale_ids.extend(
                point.id for point in points if point.id not in current_ids
            )
            if offset is None:
                break
        if stale_ids:
            client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=stale_ids),
                wait=True,
            )


def write_chunk_manifest(path: Path, chunks: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(json.dumps(chunk, ensure_ascii=True) + "\n")
    return temporary


def ingest(
    docs_dir: Path = DOCS_DIR,
    manifest_path: Path = MANIFEST_PATH,
    output_path: Path = OUTPUT_PATH,
) -> int:
    load_dotenv(BACKEND_DIR / ".env")
    documents = load_manifest(manifest_path, docs_dir)
    chunks: list[dict[str, Any]] = []
    for document in documents:
        source_file = document["source_file"]
        sections = extract_pdf_sections(
            docs_dir / source_file, document["document_title"]
        )
        document_chunks = chunk_sections(sections, document, source_file)
        if not document_chunks:
            raise ValueError(f"No text chunks generated for {source_file}.")
        chunks.extend(document_chunks)

    import os

    from qdrant_client import QdrantClient, models

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not qdrant_url or not qdrant_api_key or not openrouter_api_key:
        raise RuntimeError(
            "Set QDRANT_URL, QDRANT_API_KEY, and OPENROUTER_API_KEY in backend/.env."
        )

    openrouter = OpenRouterClient(openrouter_api_key)
    try:
        vectors = embed_chunks(openrouter, chunks)
    finally:
        openrouter.close()
    collection_name = os.getenv("QDRANT_COLLECTION", COLLECTION_NAME)
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    ensure_collection(client, collection_name)
    temporary_manifest = write_chunk_manifest(output_path, chunks)
    try:
        points = [
            models.PointStruct(
                id=qdrant_point_id(chunk["chunk_id"]),
                vector={"dense": vector},
                payload=chunk,
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        for start in range(0, len(points), EMBEDDING_BATCH_SIZE):
            client.upsert(
                collection_name=collection_name,
                points=points[start : start + EMBEDDING_BATCH_SIZE],
                wait=True,
            )
        remove_stale_points(client, collection_name, chunks)
        temporary_manifest.replace(output_path)
    except Exception:
        temporary_manifest.unlink(missing_ok=True)
        raise

    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract, chunk, embed, and index PDF documents."
    )
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    count = ingest(args.docs_dir, args.manifest, args.output)
    import os

    print(
        f"Indexed {count} chunks into {os.getenv('QDRANT_COLLECTION', COLLECTION_NAME)}."
    )


if __name__ == "__main__":
    main()
