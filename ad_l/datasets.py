"""Validated benchmark records and reproducible, balanced training pools."""

import hashlib
import json
import random
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

DATASETS = ("gsm8k", "aqua", "svamp", "multiarith")


@dataclass(frozen=True)
class Example:
    sample_id: str
    task: str
    target: str
    dataset: str = "gsm8k"
    split: str = "train"
    answer_type: str = "numeric"
    choices: tuple[str, ...] = ()


def numeric_target(value):
    try:
        number = Decimal(str(value).replace(",", "").strip())
    except InvalidOperation as error:
        raise ValueError("Expected a numeric target") from error
    if not number.is_finite():
        raise ValueError("Expected a finite numeric target")
    return str(number)


def question_fingerprint(question: str, choices=()):
    content = [" ".join(question.casefold().split()), *[" ".join(choice.casefold().split()) for choice in choices]]
    return hashlib.sha256(json.dumps(content, ensure_ascii=False).encode()).hexdigest()


def read_examples(path: Path, *, expected_split="train", dataset=None):
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    named_split = {"train": "train", "test": "test", "dev": "validation", "validation": "validation"}.get(path.stem)
    if named_split is not None and named_split != expected_split:
        raise ValueError(f"Expected {expected_split} data, received {named_split} file {path.name}")
    if expected_split == "train" and digest == "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14":
        raise ValueError("The official GSM8K test set cannot be used as training data")
    records, seen = [], set()
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            question = row["question"]
            if not isinstance(question, str) or not question.strip():
                raise ValueError("question must be nonempty")
            if "answer_type" not in row:
                name = "gsm8k"
                split = named_split or expected_split
                answer = row["answer"]
                if not isinstance(answer, str) or "####" not in answer:
                    raise ValueError("Expected a GSM8K answer with ####")
                target = numeric_target(answer.rsplit("####", 1)[1])
                answer_type, choices = "numeric", ()
                sample_id = f"gsm8k:{split}:{len(records)}"
            else:
                name, split = row["dataset"], row["split"]
                answer_type = row["answer_type"]
                sample_id = row["id"]
                options = row.get("choices", [])
                if not isinstance(options, list) or any(not isinstance(option, str) or not option.strip() for option in options):
                    raise ValueError("choices must be a list of nonempty strings")
                choices = tuple(options)
                if answer_type == "numeric":
                    target = numeric_target(row["target"])
                    if choices:
                        raise ValueError("Numeric questions cannot have choices")
                elif answer_type == "choice":
                    target = row["target"]
                    if (not 2 <= len(choices) <= 26 or not isinstance(target, str)
                            or len(target) != 1 or target not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(choices)]):
                        raise ValueError("Choice target must be one of the available option letters")
                else:
                    raise ValueError("answer_type must be numeric or choice")
            if (name not in DATASETS or (dataset is not None and name != dataset) or split != expected_split
                    or not isinstance(sample_id, str) or not sample_id or sample_id in seen):
                raise ValueError("Dataset, split or sample ID mismatch")
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path.name} row {line_number}: {error}") from error
        seen.add(sample_id)
        task = question.strip()
        if choices:
            task += "\n\nOptions\n" + "\n".join(f"{chr(65 + i)}) {option}" for i, option in enumerate(choices))
        records.append(Example(sample_id, task, target, name, split, answer_type, choices))
    if not records:
        raise ValueError("Dataset must be nonempty")
    names = {record.dataset for record in records}
    if len(names) != 1:
        raise ValueError("Each input file must contain one dataset")
    manifest_path = path.with_name("manifest.json")
    provenance = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest["splits"][expected_split]
        if entry["file"] != path.name or entry["sha256"] != digest or entry["records"] != len(records):
            raise ValueError("Dataset does not match its preparation manifest")
        provenance = {"sources": manifest["sources"], "split_policy": manifest["split_policy"]}
    return records, {"name": records[0].dataset, "path": str(path.resolve()), "split": expected_split,
                     "sha256": digest, "records": len(records), "provenance": provenance}


def load_examples(path: Path, limit: int, seed: int):
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be positive")
    records, metadata = read_examples(path)
    if limit > len(records):
        raise ValueError("limit exceeds the training dataset size")
    indices = sorted(random.Random(seed).sample(range(len(records)), limit))
    return [records[index] for index in indices], metadata


def load_training_mix(data_dir: Path, datasets, limit: int, seed: int):
    if not datasets or len(set(datasets)) != len(datasets) or any(name not in DATASETS for name in datasets):
        raise ValueError("Select distinct supported datasets")
    pools, metadata = [], {}
    for name in datasets:
        pool, info = load_examples(Path(data_dir) / name / "train.jsonl", limit, seed)
        if info["name"] != name:
            raise ValueError("Dataset directory and contents disagree")
        pools.append(pool)
        metadata[name] = info
    # Each source contributes equally, independent of its original dataset size.
    examples = [pool[index] for index in range(limit) for pool in pools]
    return examples, {"split": "train", "datasets": metadata, "selected": len(examples),
                      "sampling": {"strategy": "round_robin", "limit_per_dataset": limit, "seed": seed}}
