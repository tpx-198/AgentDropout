"""离线运行两个智能体、两轮固定通信图和最终汇总。"""

import argparse
import asyncio
import importlib.metadata as metadata
import json
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

from ad_l.fixed_graph import AGENTS, build_offline_fixed_graph, load_fixed_graph_config, resolve_fixed_topology
from ad_l.models import ModelSettings
from ad_l.prompts import load_fixed_graph_prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Anna has 3 apples and buys 2 more. How many apples does she have?")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fixed_graph_offline.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--async", dest="use_async", action="store_true")
    args = parser.parse_args()
    if not args.task.strip():
        parser.error("task 不能为空")
    try:
        config = load_fixed_graph_config(args.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    random.seed(args.seed)
    graph = build_offline_fixed_graph(config)
    started = perf_counter()
    with tracing_context(enabled=False):
        if args.use_async:
            state = asyncio.run(graph.ainvoke({"task": args.task}))
        else:
            state = graph.invoke({"task": args.task})
    record = {
        "schema_version": 1, "run_id": uuid4().hex,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "gsm8k", "task": args.task, "rounds": 2,
        "agents": AGENTS, "decision_method": "FinalRefer",
        "data_split": "manual", "sample_id": None,
        "local_seed": args.seed, "provider_seed": None,
        "settings": asdict(ModelSettings()), "config": config.to_dict(),
        "effective_topology": resolve_fixed_topology(config),
        "prompt_source": load_fixed_graph_prompts()["source"],
        "execution_mode": "async" if args.use_async else "sync",
        "round_outputs": state["round_outputs"], "trace": state["trace"],
        "answer": state["answer"], "usage": state["usage"],
        "usage_source": "mock_not_measured", "model_calls": state["model_calls"],
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
    print(f"模拟模型调用：{state['model_calls']}；真实模型请求：0；Token 用量：未测量")


if __name__ == "__main__":
    main()
