"""Prepare supervised fine-tuning data for Kubernetes assistant training.

Primary dataset:
- ComponentSoft/k8s-kubectl-cot-20k

Supplemental dataset (small sample only):
- sozercan/k8s-instructions
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

PRIMARY_DATASET = "ComponentSoft/k8s-kubectl-cot-20k"
SUPPLEMENTAL_DATASET = "sozercan/k8s-instructions"
SYSTEM_PROMPT = (
    "You are a Kubernetes and AI infrastructure assistant. "
    "Provide concise, practical, and accurate guidance."
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def _pick_split(dataset_obj: DatasetDict, dataset_name: str) -> str:
    if "train" in dataset_obj:
        return "train"
    first_split = next(iter(dataset_obj.keys()), None)
    if first_split is None:
        raise ValueError(f"No splits available for dataset: {dataset_name}")
    print(f"[WARN] '{dataset_name}' has no 'train' split. Using '{first_split}' instead.")
    return first_split


def _format_primary_record(record: dict[str, Any]) -> dict[str, str] | None:
    question = _clean_text(record.get("question"))
    command = _clean_text(record.get("command"))
    description = _clean_text(record.get("description"))
    syntax = _clean_text(record.get("syntax"))
    flags = record.get("flags")

    if not question or not command:
        return None

    prompt = (
        f"System: {SYSTEM_PROMPT}\n"
        "User: Explain and demonstrate this kubectl scenario:\n"
        f"Question: {question}\n"
        "Assistant:"
    )

    completion_lines = [
        f"Command: {command}",
    ]
    if description:
        completion_lines.append(f"Description: {description}")
    if syntax:
        completion_lines.append(f"Syntax: {syntax}")

    if flags:
        if isinstance(flags, list):
            cleaned_flags = [_clean_text(flag) for flag in flags if _clean_text(flag)]
            if cleaned_flags:
                completion_lines.append("Flags: " + "; ".join(cleaned_flags))
        elif isinstance(flags, dict):
            parts = []
            for k, v in flags.items():
                key = _clean_text(k)
                val = _clean_text(v)
                if key and val:
                    parts.append(f"{key}: {val}")
                elif key:
                    parts.append(key)
                elif val:
                    parts.append(val)
            if parts:
                completion_lines.append("Flags: " + "; ".join(parts))
        else:
            flags_text = _clean_text(flags)
            if flags_text:
                completion_lines.append(f"Flags: {flags_text}")

    return {
        "prompt": prompt,
        "completion": "\n".join(completion_lines),
        "source": PRIMARY_DATASET,
    }


def _pick_text_column(dataset: Dataset) -> str:
    candidates = [
        "instruction",
        "prompt",
        "input",
        "question",
        "query",
        "text",
        "yaml",
        "manifest",
        "output",
        "answer",
        "response",
    ]
    for name in candidates:
        if name in dataset.column_names:
            return name
    raise ValueError(
        f"Could not infer a text-like column for supplemental dataset. Columns: {dataset.column_names}"
    )


def _pick_target_column(dataset: Dataset, source_column: str) -> str:
    candidates = ["output", "answer", "response", "completion", "text", "yaml", "manifest"]
    for name in candidates:
        if name in dataset.column_names and name != source_column:
            return name
    return source_column


def _format_supplemental_record(
    record: dict[str, Any], source_column: str, target_column: str
) -> dict[str, str] | None:
    src = _clean_text(record.get(source_column))
    tgt = _clean_text(record.get(target_column))

    if not src and not tgt:
        return None

    if not src:
        src = "Generate or explain a Kubernetes manifest for the requested task."

    if not tgt:
        tgt = src

    prompt = (
        f"System: {SYSTEM_PROMPT}\n"
        "User: Kubernetes YAML / manifest request:\n"
        f"{src}\n"
        "Assistant:"
    )

    return {
        "prompt": prompt,
        "completion": tgt,
        "source": SUPPLEMENTAL_DATASET,
    }


def build_dataset(supplemental_limit: int, seed: int) -> list[dict[str, str]]:
    primary_ds = load_dataset(PRIMARY_DATASET)
    supp_ds = load_dataset(SUPPLEMENTAL_DATASET)

    primary = primary_ds[_pick_split(primary_ds, PRIMARY_DATASET)]
    supplemental = supp_ds[_pick_split(supp_ds, SUPPLEMENTAL_DATASET)]

    processed: list[dict[str, str]] = []

    for record in primary:
        item = _format_primary_record(record)
        if item:
            processed.append(item)

    source_column = _pick_text_column(supplemental)
    target_column = _pick_target_column(supplemental, source_column)

    sampled = list(supplemental)
    random.Random(seed).shuffle(sampled)
    sampled = sampled[:supplemental_limit]

    for record in sampled:
        item = _format_supplemental_record(record, source_column, target_column)
        if item:
            processed.append(item)

    return processed


def save_jsonl(records: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument(
        "--supplemental-limit",
        type=int,
        default=300,
        help="How many examples to sample from the supplemental dataset.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-samples", type=int, default=3)
    args = parser.parse_args()

    records = build_dataset(supplemental_limit=args.supplemental_limit, seed=args.seed)
    save_jsonl(records, args.output)

    print(f"Saved {len(records)} examples to {args.output}")
    print("\nSample records:")
    for sample in records[: args.show_samples]:
        print("-" * 80)
        print(json.dumps(sample, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
