"""Download and normalize the reasoning benchmarks used by AgentDropout."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
from urllib.request import Request, urlopen

from ad_l.datasets import numeric_target, question_fingerprint

SOURCES = {
    "aqua": {
        "train": "https://raw.githubusercontent.com/google-deepmind/AQuA/master/train.json",
        "validation": "https://raw.githubusercontent.com/google-deepmind/AQuA/master/dev.json",
        "test": "https://raw.githubusercontent.com/google-deepmind/AQuA/master/test.json",
        "license": "https://raw.githubusercontent.com/google-deepmind/AQuA/master/LICENSE",
    },
    "svamp": {
        "all": "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json",
        "license": "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/LICENSE",
    },
    "multiarith": {
        "all": "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/main/dataset/MultiArith/MultiArith.json",
    },
}


def grouped_split(rows: list[dict], seed: int):
    """Keep identical questions together while making a reproducible local holdout."""
    groups = defaultdict(list)
    for row in rows:
        groups[question_fingerprint(row["question"], row.get("choices", []))].append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    boundary = int(len(keys) * 0.8)
    if not 0 < boundary < len(keys):
        raise ValueError("At least two distinct questions are required")
    train_keys = set(keys[:boundary])
    splits = {"train": [], "test": []}
    for row in rows:
        split = "train" if question_fingerprint(row["question"], row.get("choices", [])) in train_keys else "test"
        splits[split].append({**row, "split": split})
    return splits


def normalize(name: str, content: dict[str, bytes], seed: int):
    if name == "aqua":
        splits = {}
        for split in ("train", "validation", "test"):
            rows = [json.loads(line) for line in content[split].decode("utf-8").splitlines() if line.strip()]
            splits[split] = [{"id": f"aqua:{split}:{index}", "dataset": name, "split": split,
                              "question": row["question"].strip(), "answer_type": "choice",
                              "choices": [re.sub(r"^[A-E]\s*\)\s*", "", option).strip() for option in row["options"]],
                              "target": row["correct"].strip().upper()}
                             for index, row in enumerate(rows)]
        held_out = {question_fingerprint(row["question"], row["choices"])
                    for split in ("test", "validation") for row in splits[split]}
        original_train = len(splits["train"])
        splits["train"] = [row for row in splits["train"]
                           if question_fingerprint(row["question"], row["choices"]) not in held_out]
        policy = {"kind": "official", "excluded_train_overlap": original_train - len(splits["train"])}
    else:
        source_rows = json.loads(content["all"])
        rows = []
        for index, row in enumerate(source_rows):
            if name == "svamp":
                question = row["Body"].strip() + " " + row["Question"].strip()
                target = row["Answer"]
            else:
                if len(row["lSolutions"]) != 1:
                    raise ValueError("MultiArith record must have one numeric solution")
                question, target = row["sQuestion"].strip(), row["lSolutions"][0]
            rows.append({"id": f"{name}:{index}", "dataset": name, "question": question,
                         "answer_type": "numeric", "target": numeric_target(target)})
        splits = grouped_split(rows, seed)
        policy = {"kind": "local_grouped_holdout", "seed": seed, "train_group_fraction": 0.8,
                  "note": "Local experimental split of the full benchmark, not the published evaluation split"}
    return splits, policy


def verify_directory(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["splits"].values():
        raw = (directory / item["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"] or len(raw.splitlines()) != item["records"]:
            raise ValueError(f"Existing data failed verification: {directory}")
    return manifest


def prepare(name: str, output_dir: Path, seed: int):
    destination = output_dir / name
    if destination.exists():
        manifest = verify_directory(destination)
        if manifest["split_policy"].get("seed", seed) != seed:
            raise ValueError("Existing dataset uses another split seed; select a new output directory")
        return manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=output_dir))
    try:
        content, sources = {}, []
        for split, url in SOURCES[name].items():
            request = Request(url, headers={"User-Agent": "AD-L benchmark preparation"})
            with urlopen(request, timeout=120) as response:
                raw = response.read()
            content[split] = raw
            sources.append({"source_split": split, "url": url, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
        splits, policy = normalize(name, content, seed)
        manifest = {"schema_version": 1, "dataset": name, "sources": sources, "split_policy": policy,
                    "prepared_at_utc": datetime.now(timezone.utc).isoformat(), "splits": {}}
        for split, rows in splits.items():
            raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
            filename = f"{split}.jsonl"
            (staging / filename).write_bytes(raw)
            manifest["splits"][split] = {"file": filename, "records": len(rows), "sha256": hashlib.sha256(raw).hexdigest()}
        if "license" in content:
            (staging / "LICENSE").write_bytes(content["license"])
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        from ad_l.datasets import read_examples

        for split in splits:
            read_examples(staging / f"{split}.jsonl", expected_split=split, dataset=name)
        os.rename(staging, destination)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def register_existing_gsm8k(output_dir: Path):
    directory = output_dir / "gsm8k"
    if not directory.exists():
        return
    expected = {
        "train": (7473, "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"),
        "test": (1319, "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"),
    }
    sources, splits = [], {}
    for split, (count, digest) in expected.items():
        raw = (directory / f"{split}.jsonl").read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest or len(raw.splitlines()) != count:
            raise ValueError("Existing GSM8K files do not match the official release")
        sources.append({"source_split": split, "url":
                        f"https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl",
                        "sha256": digest, "bytes": len(raw)})
        splits[split] = {"file": f"{split}.jsonl", "records": count, "sha256": digest}
    manifest = {"schema_version": 1, "dataset": "gsm8k", "sources": sources,
                "split_policy": {"kind": "official"}, "splits": splits}
    temporary = directory / ".manifest.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(directory / "manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", choices=tuple(SOURCES), nargs="+", default=list(SOURCES))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    register_existing_gsm8k(args.output_dir)
    for name in args.datasets:
        manifest = prepare(name, args.output_dir, args.seed)
        print(name, {split: info["records"] for split, info in manifest["splits"].items()}, flush=True)


if __name__ == "__main__":
    main()
