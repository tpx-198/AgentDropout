"""运行单智能体；默认离线，保存可复核记录。"""

import argparse
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

from ad_l.models import ModelSettings, create_model_adapter
from ad_l.prompts import load_math_solver_prompt
from ad_l.single_agent import build_single_agent_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Anna has 3 apples and buys 2 more. How many apples does she have?")
    parser.add_argument("--backend", choices=["mock", "deepseek"], default="mock")
    parser.add_argument("--allow-real-api", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mock-response", default="[MOCK: fixed response, not model reasoning]\nThe answer is 5")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.task.strip():
        parser.error("task 不能为空")
    if args.backend == "deepseek":
        if not args.allow_real_api:
            parser.error("真实模型请求需要 --allow-real-api")
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    model_name = args.model or (
        os.environ.get("DEEPSEEK_MODEL", "") if args.backend == "deepseek" else "local-fixed-response"
    )
    if not model_name.strip():
        parser.error("请通过 --model 或 DEEPSEEK_MODEL 指定模型名称")
    try:
        settings = ModelSettings(
            backend=args.backend, model=model_name,
            temperature=args.temperature, max_tokens=args.max_tokens,
        )
        adapter = create_model_adapter(
            settings, mock_response=args.mock_response, allow_real_api=args.allow_real_api,
        )
    except ValueError as error:
        parser.error(str(error))

    random.seed(args.seed)
    graph = build_single_agent_graph(adapter)
    started = perf_counter()
    with tracing_context(enabled=False):
        state = graph.invoke({"task": args.task})
    elapsed = perf_counter() - started
    record = {
        "schema_version": 1,
        "run_id": uuid4().hex,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "agent": "MathSolver", "role": "Math Solver", "domain": "gsm8k",
        "rounds": 1, "data_split": "manual", "sample_id": None,
        "local_seed": args.seed, "provider_seed": None,
        "settings": asdict(settings),
        "prompt_source": load_math_solver_prompt()["source"],
        "messages": [{"role": m.type, "content": m.content} for m in state["messages"]],
        "answer": state["answer"],
        "usage": state["usage"],
        "usage_source": "mock_not_measured" if args.backend == "mock" else (
            "provider" if state["usage"] is not None else "provider_unavailable"
        ),
        "model_calls": 1,
        "real_api_calls": int(args.backend == "deepseek"),
        "elapsed_seconds": elapsed,
        "versions": {name: metadata.version(name) for name in (
            "langchain-core", "langchain-deepseek", "langgraph", "python-dotenv",
        )},
    }
    output = args.output or ROOT / "results" / "single_agent" / f"{record['run_id']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(state["answer"])
    print(f"记录：{output.resolve()}")
    print(f"真实模型请求：{record['real_api_calls']}；Token 用量：{state['usage']}")


if __name__ == "__main__":
    main()
