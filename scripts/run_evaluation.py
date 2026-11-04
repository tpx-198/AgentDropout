"""GSM8K 批量基线：默认离线，--deepseek 明确启用真实请求。"""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ad_l.datasets import GSM8K_FILES, load_gsm8k, sample_gsm8k
from ad_l.evaluation import SCORER, run_baselines
from ad_l.fixed_graph import load_fixed_graph_config, resolve_fixed_topology
from ad_l.models import ModelSettings, create_model_adapter
from ad_l.prompts import load_fixed_graph_prompts, load_math_solver_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepseek", action="store_true", help="允许真实请求；both 模式每题最多 6 次调用")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--baseline", choices=["single", "fixed", "both"], default="both")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/gsm8k")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fixed_graph_offline.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plan", action="store_true", help="只查看样本与调用规模，不读取密钥、不创建模型")
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    try:
        pool, dataset = load_gsm8k(args.data_dir / GSM8K_FILES[args.split][0], args.split, verify_official=True)
        samples = sample_gsm8k(pool, args.limit, args.seed)
        config = load_fixed_graph_config(args.config)
    except (OSError, ValueError) as error:
        parser.error(f"{error}；数据准备：python scripts/prepare_gsm8k.py")
    baselines = ("single", "fixed") if args.baseline == "both" else (args.baseline,)
    calls = args.limit * sum(1 if baseline == "single" else 5 for baseline in baselines)
    plan = {"dataset": dataset, "sample_ids": [sample.sample_id for sample in samples],
            "seed": args.seed, "baselines": baselines, "planned_model_calls": calls,
            "planned_real_api_calls": calls if args.deepseek else 0}
    if args.plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if not args.deepseek and any(value is not None for value in (args.model, args.temperature, args.max_tokens)):
        parser.error("真实模型参数需要 --deepseek")
    factory = None
    settings = ModelSettings()
    if args.deepseek:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
        try:
            settings = ModelSettings(backend="deepseek", model=args.model or os.getenv("DEEPSEEK_MODEL", ""),
                                     temperature=args.temperature, max_tokens=args.max_tokens,
                                     timeout=args.timeout, max_retries=0)
            adapter = create_model_adapter(settings, allow_real_api=True)
        except ValueError as error:
            parser.error(str(error))
        except Exception as error:
            parser.error(f"模型客户端初始化失败：{type(error).__name__}")
        factory = lambda: adapter
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    source_files = [*sorted((ROOT / "src/ad_l").glob("*.py")),
                    *sorted((ROOT / "src/ad_l/resources/prompts").glob("*.json")), Path(__file__)]
    run_id = uuid4().hex
    manifest = {"schema_version": 1, "run_id": run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                **plan, "local_seed": args.seed, "provider_seed": None, "scorer": SCORER,
                "metrics_kind": "mock_diagnostic" if not args.deepseek else "model_evaluation",
                "settings": asdict(settings), "agents": {"single": "Math Solver", "fixed": ["Math Solver", "Mathematical Analyst", "FinalRefer"]},
                "fixed_topology": resolve_fixed_topology(config),
                "fixed_config": config.to_dict() if not args.deepseek else {
                    key: value for key, value in config.to_dict().items()
                    if key in ("rounds", "spatial_masks", "temporal_masks")},
                "prompt_sources": {"single": load_math_solver_prompt()["source"], "fixed": load_fixed_graph_prompts()["source"]},
                "code": {"git_commit": commit or None, "dirty": bool(dirty),
                         "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}},
                "python": platform.python_version(),
                "versions": {name: metadata.version(name) for name in (
                    "langchain-core", "langchain-deepseek", "langgraph", "torch", "python-dotenv")}}
    output = args.output_dir or ROOT / "results/gsm8k" / run_id
    if output.exists():
        parser.error("output-dir 已存在，请选择新目录；不会覆盖既有评测")
    print(f"计划：{args.split} {len(samples)} 题，最多 {calls} 次模型调用；真实后端：{args.deepseek}")
    report = run_baselines(samples, config, settings, output, manifest, baselines=baselines,
                           allow_real_api=args.deepseek, adapter_factory=factory)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"评测记录：{output.resolve()}")
    if report["status"] != "success":
        print("评测已停止，已保存逐题记录及部分汇总。", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
