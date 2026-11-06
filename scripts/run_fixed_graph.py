"""运行固定图，默认离线；真实模型必须显式启用。"""

import argparse
import asyncio
import importlib.metadata as metadata
import json
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langsmith import tracing_context

from ad_l.fixed_graph import build_fixed_graph, load_fixed_graph_config, resolve_fixed_topology
from ad_l.models import ModelSettings, create_model_adapter
from ad_l.prompts import load_fixed_graph_prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Anna has 3 apples and buys 2 more. How many apples does she have?")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fixed_graph_offline.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--async", dest="use_async", action="store_true")
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--backend", choices=["mock", "deepseek"], default="mock")
    backend.add_argument("--deepseek", action="store_true",
                         help="使用 DeepSeek 并允许真实请求；等同 --backend deepseek --allow-real-api")
    parser.add_argument("--allow-real-api", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.deepseek:
        args.backend = "deepseek"
        args.allow_real_api = True
    if not args.task.strip():
        parser.error("task 不能为空")
    try:
        config = load_fixed_graph_config(args.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    adapters = None
    settings = ModelSettings()
    if args.backend == "mock" and any(value is not None for value in (args.model, args.temperature, args.max_tokens)):
        parser.error("模型生成参数仅用于真实后端；模拟回答请修改 --config")
    if args.backend == "deepseek":
        if not args.allow_real_api:
            parser.error(f"真实模型请求需要 --allow-real-api；完整运行将尝试 {config.model_calls()} 次模型调用")
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
        adapters = {node_id: adapter for node_id in (*config.node_ids, "final")}
    random.seed(args.seed)
    graph = build_fixed_graph(config, adapters, allow_real_api=args.allow_real_api)
    started = perf_counter()
    with tracing_context(enabled=False):
        if args.use_async:
            state = asyncio.run(graph.ainvoke({"task": args.task}))
        else:
            state = graph.invoke({"task": args.task})
    record = {
        "schema_version": 2, "run_id": uuid4().hex,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "gsm8k", "task": args.task, "rounds": 2,
        "agents": dict(config.agents), "decision_method": "FinalRefer",
        "data_split": "manual", "sample_id": None,
        "local_seed": args.seed, "provider_seed": None,
        "settings": asdict(settings), "config": config.to_dict() if args.backend == "mock" else {
            key: value for key, value in config.to_dict().items()
            if key in ("rounds", "spatial_masks", "temporal_masks")},
        "effective_topology": resolve_fixed_topology(config),
        "prompt_source": load_fixed_graph_prompts()["source"],
        "execution_mode": "async" if args.use_async else "sync",
        "round_outputs": state["round_outputs"], "trace": state["trace"],
        "answer": state["answer"], "usage": state["usage"],
        "status": state["status"], "known_usage": state["known_usage"],
        "setup_error": state.get("setup_error"),
        "usage_complete": state["usage_complete"],
        "usage_source": "mock_not_measured" if args.backend == "mock" else (
            "provider" if state["usage_complete"] else "provider_incomplete"),
        "model_calls": state["model_calls"],
        "real_api_calls": state["real_api_calls"],
        "elapsed_seconds": perf_counter() - started,
        "versions": {name: metadata.version(name) for name in (
            "langchain-core", "langchain-deepseek", "langgraph", "python-dotenv",
        )},
    }
    output = args.output or ROOT / "results/fixed_graph" / f"{record['run_id']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for call in state["trace"]:
        print(f"第 {call['round']} 轮 {call['node_id']}："
              f"轮内可见 {list(call['spatial_info'])}，跨轮可见 {list(call['temporal_info'])}")
    print(state["answer"])
    print(f"记录：{output.resolve()}")
    print(f"状态：{state['status']}；模型调用尝试：{state['model_calls']}；"
          f"真实后端调用尝试：{state['real_api_calls']}；Token 用量：{state['usage']}")
    if state["status"] == "failed":
        print("模型调用失败，已保存之前的记录和失败类型。", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
