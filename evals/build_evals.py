"""Validate and export the canonical evaluation dataset.

The gold dataset is maintained in eval_dataset.json. This script keeps the
JSONL form synchronized for line-oriented tooling without deriving labels from
the source corpus at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
JSON_PATH = ROOT / "eval_dataset.json"
JSONL_PATH = ROOT / "eval_dataset.jsonl"
REQUIRED_FIELDS = {
    "id",
    "category",
    "difficulty",
    "question",
    "gold_answer",
    "key_facts",
    "answerable",
    "source_sections",
    "source_docs",
    "distractor_docs",
    "notes",
}


def validate(questions: list[dict]) -> None:
    ids = [question.get("id") for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation question IDs must be unique.")
    for question in questions:
        missing = REQUIRED_FIELDS - question.keys()
        if missing:
            raise ValueError(
                f"{question.get('id', '<unknown>')} missing {sorted(missing)}"
            )
        if not isinstance(question["answerable"], bool):
            raise ValueError(f"{question['id']} answerable must be boolean")


def main() -> None:
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    questions = payload["questions"]
    validate(questions)
    payload["meta"]["num_questions"] = len(questions)
    JSON_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    JSONL_PATH.write_text(
        "".join(
            json.dumps(question, ensure_ascii=False) + "\n" for question in questions
        ),
        encoding="utf-8",
    )
    print(f"validated {len(questions)} evaluation questions")


if __name__ == "__main__":
    main()
