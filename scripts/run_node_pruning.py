"""训练节点选择权重并配对评测节点剪枝，默认使用本地模拟。"""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import torch

from ad_l.datasets import load_gsm8k, sample_gsm8k
from ad_l.evaluation import SCORER, run_baselines, summarize, write_json
from ad_l.fixed_graph import load_fixed_graph_config
from ad_l.models import ModelSettings, create_model_adapter
from ad_l.node_pruning import fixed_degrees, node_pruned_masks, prune_nodes, train_selector_step
from ad_l.pruning import compare_pairs
from ad_l.topology import TopologyPolicy

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=3, help="参与配对评测的 test 题数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare-random", action="store_true", help="增加同调用预算的随机节点剪枝对照")
    parser.add_argument("--random-seed", type=int, help="随机对照的独立种子，默认与 --seed 数值相同")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--initial-probability", type=float, default=0.5)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fixed_graph_offline.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/gsm8k")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--deepseek", action="store_true", help="显式启用真实模型调用")
    args = parser.parse_args()
    if args.random_seed is not None and not args.compare_random:
        parser.error("--random-seed 需要 --compare-random")
    try:
        if args.steps < 1 or args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0:
            raise ValueError("steps、batch-size 和 lr 必须为正数")
        config = load_fixed_graph_config(args.config)
        fixed_degrees(config)
        selector = TopologyPolicy(config.spatial_masks, config.temporal_masks, args.initial_probability,
                                  diff=True, node_ids=config.node_ids)
        train, train_data = load_gsm8k(args.data_dir / "train.jsonl", "train", verify_official=True)
        test, test_data = load_gsm8k(args.data_dir / "test.jsonl", "test", verify_official=True)
        train = sample_gsm8k(train, args.steps * args.batch_size, args.seed)
        test = sample_gsm8k(test, args.limit, args.seed)
        generator = torch.Generator().manual_seed(args.seed)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    full_calls = config.model_calls()
    pruned_calls = full_calls - 2
    baselines = ("before", "random", "after") if args.compare_random else ("before", "after")
    eval_budgets = {name: len(test) * (full_calls if name == "before" else pruned_calls)
                    for name in baselines}
    eval_calls = sum(eval_budgets.values())
    calls = len(train) * pruned_calls + eval_calls
    plan = {"train_dataset": train_data, "test_dataset": test_data,
            "train_sample_ids": [sample.sample_id for sample in train],
            "test_sample_ids": [sample.sample_id for sample in test],
            "hyperparameters": {"steps": args.steps, "batch_size": args.batch_size, "seed": args.seed,
                                "lr": args.lr, "initial_probability": args.initial_probability},
            "planned_training_model_calls": len(train) * pruned_calls,
            "planned_evaluation_model_calls_by_group": eval_budgets,
            "planned_model_calls": calls, "planned_real_api_calls": calls if args.deepseek else 0}
    if args.compare_random:
        seed = args.seed if args.random_seed is None else args.random_seed
        rng = random.Random(seed)
        random_plan = tuple(rng.choice(config.node_ids) for _ in range(2))
        spatial, temporal = node_pruned_masks(config, random_plan)
        random_policy = TopologyPolicy(spatial, temporal, diff=True, node_ids=config.node_ids)
        plan["random_pruning"] = {"seed": seed, "method": "uniform_independent_per_round",
                                  "skip_plan": random_plan, "spatial_masks": spatial, "temporal_masks": temporal,
                                  "effective_topology": random_policy.fixed_round_topologies(config)}
    if args.plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    output = args.output_dir or ROOT / "results/node_pruning" / uuid4().hex
    if output.exists():
        parser.error("output-dir 已存在，不覆盖现有实验")
    adapters, factory, settings = None, None, ModelSettings()
    if args.deepseek:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
        try:
            settings = ModelSettings(backend="deepseek", model=os.environ.get("DEEPSEEK_MODEL", ""))
            adapter = create_model_adapter(settings, allow_real_api=True)
        except Exception as error:
            parser.error(f"DeepSeek 初始化失败（{type(error).__name__}）；请检查 .env 中的模型和密钥配置")
        adapters = {node: adapter for node in (*config.node_ids, "final")}
        factory = lambda: adapter
    sources = [*sorted((ROOT / "src/ad_l").glob("*.py")),
               *sorted((ROOT / "src/ad_l/resources/prompts").glob("*.json")), Path(__file__).resolve()]
    manifest = {**plan, "schema_version": 1, "settings": asdict(settings), "config": config.to_dict(),
                "scorer": SCORER, "rule": "agentdropout_node_selector_v1",
                "versions": {name: metadata.version(name) for name in ("torch", "langgraph", "langchain-core", "langchain-deepseek")},
                "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}
    output.mkdir(parents=True)
    write_json(output / "manifest.json", manifest)
    print(f"节点剪枝记录：{output.resolve()}；计划最多 {calls} 次模型调用")
    optimizer = torch.optim.Adam(selector.parameters(), lr=args.lr)
    reports, episodes = [], []
    for start in range(0, len(train), args.batch_size):
        report = train_selector_step(selector, optimizer, config, train[start:start + args.batch_size],
                                     generator=generator, adapters=adapters, allow_real_api=args.deepseek)
        reports.append({"step": len(reports) + 1, **report})
        episodes.extend(report["episodes"])
        stats = summarize(episodes, ("selector",), len(train), settings.backend)["selector"]
        status = report["status"]
        if status == "success" and len(reports) < args.steps:
            status = "running"
        write_json(output / "training.json", {"status": status, "steps": reports, "summary": stats})
        if report["status"] != "success":
            print(f"节点选择训练失败；真实调用尝试 {stats['real_api_calls']} 次", file=sys.stderr)
            raise SystemExit(1)

    before, after, skip_plan, pruning = prune_nodes(selector, config)
    write_json(output / "pruning.json", pruning)
    policies = {"before": before, "after": after}
    skip_variants = {"after": skip_plan}
    if args.compare_random:
        policies["random"] = random_policy
        skip_variants["random"] = random_plan
    result = run_baselines(test, config, settings, output / "evaluation",
        {**manifest, "pruning": pruning, "inference_mode": "original-fixed",
         "planned_model_calls": eval_calls, "planned_real_api_calls": eval_calls if args.deepseek else 0},
        baselines=baselines, allow_real_api=args.deepseek, adapter_factory=factory,
        learned_mode="original-fixed", policy_variants=policies, skip_variants=skip_variants)
    rows = [json.loads(line) for line in (output / "evaluation/predictions.jsonl").read_text().splitlines()]
    evaluation = result["summary"]
    real_calls = stats["real_api_calls"] + sum(group["real_api_calls"] for group in evaluation.values())
    comparison = {"status": result["status"], "summary": evaluation,
                  "comparison": compare_pairs(rows, len(test), settings.backend), "real_api_calls": real_calls}
    if args.compare_random:
        comparison["random_pruning"] = {**plan["random_pruning"], "matches_learned_plan": random_plan == skip_plan}
        comparison["random_comparisons"] = {
            "random_vs_full": compare_pairs(rows, len(test), settings.backend, after="random"),
            "learned_vs_random": compare_pairs(rows, len(test), settings.backend, before="random"),
        }
    write_json(output / "comparison.json", comparison)
    print(f"各轮跳过：{', '.join(skip_plan)}；评测调用前后：{evaluation['before']['model_calls']} → "
          f"{evaluation['after']['model_calls']}；全程真实调用尝试：{real_calls}")
    if args.compare_random:
        print(f"随机对照各轮跳过：{', '.join(random_plan)}；评测调用：{evaluation['random']['model_calls']}")
    if result["status"] != "success":
        raise SystemExit(1)
