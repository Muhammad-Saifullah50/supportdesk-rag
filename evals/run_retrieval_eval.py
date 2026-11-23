"""Run a deterministic BM25 retrieval baseline over the local chunk manifest."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "backend" / "data" / "chunks.jsonl"
DEFAULT_DATASET = ROOT / "evals" / "eval_dataset.json"


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def expected_sections(entry: dict) -> set[tuple[str, str]]:
    sections = set()
    for reference in entry.get("source_sections", []):
        filename, section = reference.rsplit("#", 1)
        sections.add((filename, section))
    return sections


def section_key(chunk: dict) -> tuple[str, str]:
    title = chunk.get("section_title", "")
    number = title.split(".", 1)[0] if "." in title else ""
    return chunk.get("source_file", ""), number


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(ROOT / "backend"))
    from rag import bm25_rank

    chunks = load_jsonl(args.chunks)
    questions = json.loads(args.dataset.read_text(encoding="utf-8"))["questions"]
    rows = []
    for entry in questions:
        if not entry["answerable"]:
            continue
        ranked = bm25_rank(entry["question"], chunks, limit=args.k)
        expected_docs = set(entry["source_docs"])
        expected_secs = expected_sections(entry)
        ranked_docs = {chunk.get("source_file") for chunk in ranked}
        ranked_secs = {section_key(chunk) for chunk in ranked}
        first = ranked[0] if ranked else {}
        first_rank = next(
            (
                index + 1
                for index, chunk in enumerate(ranked)
                if section_key(chunk) in expected_secs
            ),
            None,
        )
        rows.append(
            {
                "id": entry["id"],
                "category": entry["category"],
                "section_hit": bool(expected_secs & ranked_secs),
                "document_hit": bool(expected_docs & ranked_docs),
                "mrr": 1 / first_rank if first_rank else 0.0,
                "stale_top1": first.get("source_file")
                in set(entry.get("distractor_docs", [])),
            }
        )

    def average(key: str) -> float:
        return round(statistics.mean(row[key] for row in rows), 4) if rows else 0.0

    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    summary = {
        "questions": len(rows),
        f"section_recall@{args.k}": average("section_hit"),
        f"document_recall@{args.k}": average("document_hit"),
        "mrr": average("mrr"),
        "stale_top1_rate": average("stale_top1"),
        "by_category": {
            category: {
                "questions": len(category_rows),
                f"section_recall@{args.k}": round(
                    statistics.mean(row["section_hit"] for row in category_rows), 4
                ),
                "mrr": round(statistics.mean(row["mrr"] for row in category_rows), 4),
            }
            for category, category_rows in sorted(by_category.items())
        },
    }
    report = {
        "dataset": str(args.dataset.relative_to(ROOT)),
        "summary": summary,
        "rows": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
