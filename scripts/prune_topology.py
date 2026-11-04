"""独立边剪枝与剪枝前后配对评测，默认离线。"""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import io
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from ad_l.datasets import load_gsm8k, sample_gsm8k
from ad_l.evaluation import SCORER, run_baselines, write_json
from ad_l.fixed_graph import OfflineFixedGraphConfig
from ad_l.models import ModelSettings, create_model_adapter
from ad_l.pruning import compare_pairs, prune_checkpoint, verify_pair
from ad_l.training import load_training_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prune-only", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--deepseek", action="store_true", help="允许真实配对评测，每题最多 10 次请求")
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/gsm8k")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        source_bytes = args.checkpoint.read_bytes()
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        checkpoint = load_training_checkpoint(io.BytesIO(source_bytes))
        config = OfflineFixedGraphConfig.from_dict(checkpoint["context"]["config"])
        pruned, pruning, before, after = prune_checkpoint(checkpoint, config, args.rate, parent_sha256=source_hash)
        verify_pair(checkpoint, pruned, source_hash)
        samples, dataset = [], None
        if not args.prune_only:
            pool, dataset = load_gsm8k(args.data_dir / "test.jsonl", "test", verify_official=True)
            samples = sample_gsm8k(pool, args.limit, args.seed)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    calls = len(samples) * 10
    if args.plan:
        print(json.dumps({"pruning": pruning, "sample_ids": [sample.sample_id for sample in samples],
                          "planned_model_calls": calls, "planned_real_api_calls": calls if args.deepseek else 0},
                         ensure_ascii=False, indent=2))
        return
    output = args.output_dir or ROOT / "results/pruning" / uuid4().hex
    if output.exists():
        parser.error("output-dir 已存在，不覆盖现有实验")
    if args.prune_only and (args.deepseek or any(value is not None for value in (args.model, args.temperature, args.max_tokens))):
        parser.error("--prune-only 不使用模型，请移除模型相关参数")
    if not args.deepseek and any(value is not None for value in (args.model, args.temperature, args.max_tokens)):
        parser.error("真实模型参数需要 --deepseek")
    settings, factory = ModelSettings(), None
    if args.deepseek:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
        try:
            settings = ModelSettings(backend="deepseek", model=args.model or os.getenv("DEEPSEEK_MODEL", ""),
                temperature=args.temperature, max_tokens=args.max_tokens, timeout=args.timeout, max_retries=0)
            adapter = create_model_adapter(settings, allow_real_api=True)
        except ValueError as error:
            parser.error(str(error))
        except Exception as error:
            parser.error(f"模型初始化失败：{type(error).__name__}")
        factory = lambda: adapter
    output.mkdir(parents=True)
    temporary = output / "checkpoint.tmp"
    torch.save(pruned, temporary)
    temporary.replace(output / "checkpoint.pt")
    write_json(output / "pruning.json", pruning)
    for kind, record in pruning["changes"].items():
        print(f"{kind}: 候选边 {record['active_before']} → {record['active_after']}，实际剪除 {record['actual_removed']}")
    if args.prune_only:
        print(f"剪枝记录：{output.resolve()}；真实请求：0")
        return
    manifest = {"schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset, "sample_ids": [sample.sample_id for sample in samples], "seed": args.seed,
        "provider_seed": None, "scorer": SCORER, "baselines": ["before", "after"],
        "inference_mode": "original-fixed", "settings": asdict(settings),
        "metrics_kind": "mock_diagnostic" if settings.backend == "mock" else "model_evaluation",
        "parent_checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": source_hash},
        "pruned_checkpoint": {"path": str((output / "checkpoint.pt").resolve()),
                              "sha256": hashlib.sha256((output / "checkpoint.pt").read_bytes()).hexdigest()},
        "training_settings": checkpoint["context"]["settings"], "pruning": pruning,
        "planned_model_calls": calls, "planned_real_api_calls": calls if args.deepseek else 0,
        "versions": {name: metadata.version(name) for name in ("torch", "langchain-core", "langchain-deepseek", "langgraph")},
        "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted((ROOT / "src/ad_l").glob("*.py"))}}
    result = run_baselines(samples, config, settings, output / "evaluation", manifest,
        baselines=("before", "after"), allow_real_api=args.deepseek, adapter_factory=factory,
        learned_mode="original-fixed", policy_variants={"before": before, "after": after})
    rows = [json.loads(line) for line in (output / "evaluation/predictions.jsonl").read_text().splitlines()]
    comparison = {"status": result["status"], "summary": result["summary"],
                  "comparison": compare_pairs(rows, len(samples), settings.backend),
                  "real_api_calls": sum(group["real_api_calls"] for group in result["summary"].values())}
    write_json(output / "comparison.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"剪枝与对照记录：{output.resolve()}")
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
