"""Evaluate the deployed SupportDesk-RAG API against the versioned dataset.

Example:
    python evals/run_live_eval.py \
        --base-url https://supportdesk-backend-muhammadsaifullah50s-projects.vercel.app
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evals" / "eval_dataset.json"
REFUSAL = "I do not have sufficient information in the provided corpus to answer this question."
NUMBER_PATTERN = re.compile(r"\$?\d[\d,.]*%?")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def load_dataset(path: Path) -> list[dict]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))["questions"]
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def fact_is_present(answer: str, fact: str) -> bool:
    answer_lower = answer.casefold()
    numbers = [value.casefold() for value in NUMBER_PATTERN.findall(fact)]
    if numbers and not all(value in answer_lower for value in numbers):
        return False

    words = [word.casefold() for word in TOKEN_PATTERN.findall(fact) if len(word) > 3]
    if not words:
        return bool(numbers)
    return sum(word in answer_lower for word in words) >= max(1, len(words) // 2)


def score_facts(answer: str, facts: list[str]) -> float:
    if not facts:
        return 1.0
    return sum(fact_is_present(answer, fact) for fact in facts) / len(facts)


def expected_documents(entry: dict) -> set[str]:
    return {Path(value).name for value in entry.get("source_docs", [])}


def request_answer(base_url: str, question: str, test_ip: str, timeout: float) -> dict:
    body = json.dumps({"question": question}).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/api/ask",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Forwarded-For": test_ip,
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def evaluate_entry(
    base_url: str,
    entry: dict,
    index: int,
    timeout: float,
) -> dict:
    started = time.perf_counter()
    try:
        payload = request_answer(
            base_url,
            entry["question"],
            f"198.51.100.{(index % 254) + 1}",
            timeout,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        answer = payload.get("answer", "")
        citations = payload.get("citations", [])
        citation_valid = all(
            isinstance(citation, dict)
            and all(
                citation.get(field)
                for field in ("document_id", "section", "pdf_filename")
            )
            for citation in citations
        )
        cited_documents = {
            Path(citation["pdf_filename"]).name
            for citation in citations
            if isinstance(citation, dict) and citation.get("pdf_filename")
        }
        expected_docs = expected_documents(entry)
        result = {
            "id": entry["id"],
            "category": entry["category"],
            "answerable": entry["answerable"],
            "status": "ok",
            "latency_ms": elapsed_ms,
            "citation_valid": citation_valid,
            "citation_count": len(citations),
            "expected_document_coverage": (
                len(expected_docs & cited_documents) / len(expected_docs)
                if expected_docs
                else 1.0
            ),
            "answer": answer,
        }
        if entry["answerable"]:
            result["fact_recall"] = score_facts(answer, entry.get("key_facts", []))
            result["all_key_facts_present"] = result["fact_recall"] == 1.0
        else:
            result["exact_refusal"] = answer == REFUSAL
        return result
    except HTTPError as exc:
        return {
            "id": entry["id"],
            "category": entry["category"],
            "answerable": entry["answerable"],
            "status": "http_error",
            "http_status": exc.code,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    except (URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
        return {
            "id": entry["id"],
            "category": entry["category"],
            "answerable": entry["answerable"],
            "status": "client_error",
            "error": str(exc),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }


def summarize(results: list[dict]) -> dict:
    successful = [result for result in results if result["status"] == "ok"]
    answerable = [result for result in successful if result["answerable"]]
    unanswerable = [result for result in successful if not result["answerable"]]
    latencies = [result["latency_ms"] for result in successful]

    def average(rows: list[dict], key: str) -> float:
        values = [row[key] for row in rows if key in row]
        return round(statistics.mean(values), 4) if values else 0.0

    by_category: dict[str, list[dict]] = defaultdict(list)
    for result in successful:
        by_category[result["category"]].append(result)

    category_metrics = {}
    for category, rows in sorted(by_category.items()):
        category_metrics[category] = {
            "count": len(rows),
            "fact_recall": average(rows, "fact_recall"),
            "exact_refusal_rate": average(rows, "exact_refusal"),
            "citation_valid_rate": average(
                [{"value": float(row["citation_valid"])} for row in rows], "value"
            ),
        }

    return {
        "total": len(results),
        "successful": len(successful),
        "errors": len(results) - len(successful),
        "answerable_fact_recall": average(answerable, "fact_recall"),
        "answerable_all_key_facts_rate": average(
            [{"value": float(row["all_key_facts_present"])} for row in answerable],
            "value",
        ),
        "unanswerable_exact_refusal_rate": average(unanswerable, "exact_refusal"),
        "citation_valid_rate": average(
            [{"value": float(row["citation_valid"])} for row in successful], "value"
        ),
        "expected_document_coverage": average(successful, "expected_document_coverage"),
        "latency_ms": {
            "average": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "p50": round(statistics.median(latencies), 2) if latencies else 0.0,
            "p95": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 2)
            if latencies
            else 0.0,
        },
        "by_category": category_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    results = [
        evaluate_entry(args.base_url, entry, index, args.timeout)
        for index, entry in enumerate(dataset)
    ]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url.rstrip("/"),
        "dataset": str(args.dataset.relative_to(ROOT)),
        "dataset_count": len(dataset),
        "summary": summarize(results),
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
