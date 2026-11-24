"""Run a configurable multi-agent system. Defaults to local mock responses."""

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import torch

from ad_l.cli import add_arguments, adapters_from_args, print_plan, resolve_policy, write_json
from ad_l.system import build_system
from ad_l.model_config import load_agent_config, model_call_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--task", default="Anna has 3 apples and buys 2 more. How many apples does she have?")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--mode", choices=("sampled", "fixed", "threshold"), default="sampled")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--prune-rate", type=float, default=0.0)
    parser.add_argument("--async", dest="use_async", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional JSON trace and weight report")
    args = parser.parse_args()
    try:
        if not args.task.strip():
            raise ValueError("task must be nonempty")
        if args.output is not None and args.output.exists():
            raise ValueError("output already exists; choose a new file")
        policy, team = resolve_policy(args, args.checkpoint)
        agent_config = load_agent_config(args.api_config, team) if args.api_config else None
        node_pruning = policy.prune_nodes(args.node_prune_count)
        pruning = policy.prune(args.prune_rate)
        generator = torch.Generator().manual_seed(args.seed)
        with torch.no_grad():
            sampled = policy.sample(generator, mode=args.mode, threshold=args.threshold,
                                    temperature=args.sampling_temperature)
        plan = print_plan(policy, team)
        plan.update(planned_real_api_calls=plan["planned_model_calls"] if args.deepseek else 0,
                    execution=sampled.execution, pruning=pruning, node_pruning=node_pruning)
        calls = plan["planned_model_calls_by_agent"]
        if agent_config is not None:
            plan.update(model_call_plan(agent_config, team, policy.config.rounds, 1, calls_by_node=calls))
        if args.plan:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        adapters, settings = adapters_from_args(args, team, agent_config,
                                                active_nodes={node for node, count in calls.items() if count})
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    except Exception as error:
        parser.error(f"Initialization failed ({type(error).__name__})")
    graph = build_system(team, sampled.execution, adapters, allow_real_api=args.deepseek or args.allow_real_api)
    state = asyncio.run(graph.ainvoke({"task": args.task})) if args.use_async else graph.invoke({"task": args.task})
    if args.output is not None:
        write_json(args.output, {"schema_version": 1, **plan, **policy.report(), "task": args.task,
                                "settings": {node: asdict(value) for node, value in settings.items()}, "result": state})
    print(state["answer"])
    print(f"Status {state['status']} | agents {len(team)} | calls {state['model_calls']} | real calls {state['real_api_calls']}")
    print(f"Token usage {state['usage']}")
    if state["status"] != "success":
        print(f"Failure type {state['trace'][-1]['error_type']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
