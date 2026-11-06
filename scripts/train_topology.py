"""GSM8K train 多步训练；默认模拟，显式 --deepseek 才请求真实模型。"""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ad_l.datasets import load_gsm8k, sample_gsm8k
from ad_l.evaluation import SCORER
from ad_l.fixed_graph import OfflineFixedGraphConfig, load_fixed_graph_config
from ad_l.models import ModelSettings, create_model_adapter
from ad_l.node_pruning import load_node_pruning
from ad_l.training import load_training_checkpoint, train_multistep, pruning_schedule, pruning_due


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, help="目标总步数；新训练默认 3，续训默认再走 1 步")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int, help="训练样本池大小，默认完整训练集")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--initial-probability", type=float)
    parser.add_argument("--temperature", type=float, help="拓扑采样温度")
    parser.add_argument("--delta", type=float)
    parser.add_argument("--prune-every", type=int, help="每 N 个成功更新步后剪枝；默认 0 表示关闭")
    parser.add_argument("--prune-rate", type=float, help="每次剪枝比例，默认 0.25")
    parser.add_argument("--prune-until", type=int, help="允许剪枝的最后一个绝对训练步（含）")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--node-pruning", type=Path, help="从 nodes.py 的 pruning.json 开始边训练，自动开启 --diff")
    parser.add_argument("--diff", action=argparse.BooleanOptionalAction, default=None,
                        help="每轮独立拓扑参数；默认共享，续训自动继承")
    parser.add_argument("--deepseek", action="store_true", help="允许真实训练，可用 --plan 查看调用上限")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--model-temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/gsm8k")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        checkpoint = load_training_checkpoint(args.resume) if args.resume else None
        old = checkpoint["context"] if checkpoint else {}
        if checkpoint and args.node_pruning:
            raise ValueError("续训从检查点继承节点剪枝计划")
        diff = old.get("diff", bool(args.node_pruning)) if args.diff is None else args.diff
        if checkpoint and diff != old.get("diff", False):
            raise ValueError("续训不能切换 diff 参数模式")
        defaults = {"seed": 0, "lr": 0.1, "batch_size": 2, "initial_probability": 0.5,
                    "temperature": 1.0, "delta": 0.1, "limit": None,
                    "prune_every": 0, "prune_rate": 0.25, "prune_until": None}
        hp = {}
        for name, default in defaults.items():
            value = getattr(args, name)
            saved = old.get("hyperparameters", {}).get(name, default)
            if checkpoint and value is not None and value != saved:
                raise ValueError(f"续训不能更改 {name}")
            hp[name] = saved if value is None else value
        schedule = pruning_schedule({"hyperparameters": hp})
        if (hp["batch_size"] < 1 or not math.isfinite(hp["lr"]) or hp["lr"] <= 0
                or not math.isfinite(hp["initial_probability"]) or not 0 < hp["initial_probability"] < 1
                or not math.isfinite(hp["temperature"]) or hp["temperature"] <= 0
                or not math.isfinite(hp["delta"]) or hp["delta"] < 0):
            raise ValueError("批量、学习率、初始概率、拓扑温度或 delta 不合法")
        pool, dataset = load_gsm8k(args.data_dir / "train.jsonl", "train", verify_official=True)
        if hp["limit"] is not None:
            pool = sample_gsm8k(pool, hp["limit"], hp["seed"])
        if checkpoint and args.config is None:
            config = OfflineFixedGraphConfig.from_dict(old["config"])
        else:
            config = load_fixed_graph_config(args.config or ROOT / "configs/fixed_graph_offline.json")
        node_pruning = load_node_pruning(args.node_pruning, config) if args.node_pruning else old.get("node_pruning")
        if node_pruning is not None and not diff:
            raise ValueError("节点剪枝只支持逐轮独立边拓扑")
        start = checkpoint["step"] if checkpoint else 0
        steps = args.steps if args.steps is not None else (start + 1 if checkpoint else 3)
        if steps <= start:
            raise ValueError("steps 必须大于已完成总步数")
        backend = "deepseek" if args.deepseek else "mock"
        if checkpoint and backend != old["settings"]["backend"]:
            raise ValueError("续训必须使用原后端；真实检查点需显式 --deepseek")
        if checkpoint and checkpoint.get("evaluation_only"):
            raise ValueError("独立剪枝检查点仅供评测；周期训练应从训练检查点续训")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    skip_nodes = node_pruning["skip_plan"] if node_pruning is not None else (None, None)
    calls = (steps - start) * hp["batch_size"] * config.model_calls(skip_nodes)
    if args.plan:
        print(json.dumps({"split": "train", "pool_size": len(pool), "step_start": start,
                          "target_steps": steps, "batch_size": hp["batch_size"],
                          "diff": diff,
                          "node_pruning": node_pruning,
                          "pruning_schedule": schedule,
                          "pruning_steps_this_run": [step for step in range(start + 1, steps + 1) if pruning_due(step, schedule)],
                          "planned_model_calls": calls, "planned_real_api_calls": calls if args.deepseek else 0}, indent=2))
        return
    factory = None
    settings = ModelSettings()
    if not args.deepseek and any(value is not None for value in (args.model, args.model_temperature, args.max_tokens, args.timeout)):
        parser.error("模型参数需要 --deepseek")
    if args.deepseek:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
        previous = old.get("settings", {})
        try:
            settings = ModelSettings(backend="deepseek", model=args.model or previous.get("model") or os.getenv("DEEPSEEK_MODEL", ""),
                temperature=args.model_temperature if args.model_temperature is not None else previous.get("temperature"),
                max_tokens=args.max_tokens if args.max_tokens is not None else previous.get("max_tokens"),
                timeout=args.timeout if args.timeout is not None else previous.get("timeout", 60.0), max_retries=0)
            adapter = create_model_adapter(settings, allow_real_api=True)
        except ValueError as error:
            parser.error(str(error))
        except Exception as error:
            parser.error(f"模型初始化失败：{type(error).__name__}")
        factory = lambda: adapter
    sources = [*sorted((ROOT / "src/ad_l").glob("*.py")),
               *sorted((ROOT / "src/ad_l/resources/prompts").glob("*.json"))]
    context = {"dataset": dataset, "pool_sample_ids": [sample.sample_id for sample in pool],
               "hyperparameters": hp, "settings": asdict(settings), "config": config.to_dict(),
               "scorer": SCORER, "diff": diff, "python": platform.python_version(),
               "versions": {name: metadata.version(name) for name in ("torch", "langgraph", "langchain-core", "langchain-deepseek")},
               "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}
    if node_pruning is not None:
        context["node_pruning"] = node_pruning
    output = args.output_dir or ROOT / "results/topology" / uuid4().hex
    try:
        report = train_multistep(pool, config, settings, context, output, steps, checkpoint=checkpoint,
                                adapter_factory=factory, allow_real_api=args.deepseek)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(f"完成总步数：{report['completed_steps']}；本次真实请求尝试：{report['real_api_calls']}")
    print(f"训练记录：{output.resolve()}")
    if report["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
