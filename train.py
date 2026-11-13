"""Train and prune edge probabilities on selected or mixed reasoning benchmarks."""

import argparse
import json
import math
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import torch

from ad_l.cli import ROOT, add_arguments, adapters_from_args, print_plan, resolve_policy
from ad_l.roles import PROMPTS
from ad_l.topology import save_policy
from ad_l.training import load_examples, train_step
from ad_l.datasets import DATASETS, load_training_mix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    data_source = parser.add_mutually_exclusive_group()
    data_source.add_argument("--data", type=Path, help="One legacy GSM8K or normalized training JSONL file")
    data_source.add_argument("--datasets", choices=DATASETS, nargs="+", help="Datasets to mix; default all four")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--limit", type=int, default=6, help="Selected training questions per dataset")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--sparsity", type=float, default=0.01)
    parser.add_argument("--prune-rate", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/topology.pt")
    args = parser.parse_args()
    try:
        if args.steps < 1 or args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0:
            raise ValueError("steps, batch-size and lr must be positive")
        if not math.isfinite(args.sparsity) or args.sparsity < 0:
            raise ValueError("sparsity must be finite and nonnegative")
        if not math.isfinite(args.sampling_temperature) or args.sampling_temperature <= 0:
            raise ValueError("sampling-temperature must be finite and positive")
        if not 0 <= args.prune_rate <= 1:
            raise ValueError("prune-rate must be in [0, 1]")
        if args.output.exists():
            raise ValueError("output already exists; choose a new file")
        policy, team = resolve_policy(args)
        if args.data is not None:
            examples, dataset = load_examples(args.data, args.limit, args.seed)
        else:
            examples, dataset = load_training_mix(args.data_dir, args.datasets or DATASETS, args.limit, args.seed)
        plan = print_plan(policy, team, args.steps * args.batch_size)
        plan["planned_real_api_calls"] = plan["planned_model_calls"] if args.deepseek else 0
        plan["dataset"] = dataset
        plan["sample_ids"] = [example.sample_id for example in examples]
        if args.plan:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        adapters, settings = adapters_from_args(args, team)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    except Exception as error:
        parser.error(f"Initialization failed ({type(error).__name__})")
    generator = torch.Generator().manual_seed(args.seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    history = []
    metadata = {**plan, "settings": asdict(settings), "roles": [agent.role for agent in team[:-1]],
                "hyperparameters": {name: getattr(args, name) for name in (
                    "seed", "steps", "batch_size", "lr", "sparsity", "prune_rate", "sampling_temperature")},
                "metrics_kind": "mock_diagnostic" if settings.backend == "mock" else "model_training",
                "prompt_source": PROMPTS["source"], "history": history, "pruning": None,
                "versions": {name: version(name) for name in ("torch", "langgraph", "langchain-core", "langchain-deepseek")}}
    for step in range(args.steps):
        batch = [examples[(step * args.batch_size + offset) % len(examples)] for offset in range(args.batch_size)]
        report = train_step(policy, optimizer, batch, adapters, generator, sparsity=args.sparsity,
                            temperature=args.sampling_temperature, allow_real_api=args.deepseek,
                            roles=[agent.role for agent in team[:-1]])
        history.append({"step": step + 1, **report})
        real_calls = sum(episode["result"]["real_api_calls"] for item in history for episode in item["episodes"])
        metadata.update(real_api_calls=real_calls, completed_steps=step + int(report["status"] == "success"),
                        status="running" if report["status"] == "success" else "failed")
        save_policy(args.output, policy, metadata)
        print(f"Step {step + 1}/{args.steps} | status {report['status']} | loss {report.get('loss')} | real calls {real_calls}")
        if report["status"] != "success":
            raise SystemExit("Model execution failed; the current batch was not updated")
    pruning = policy.prune(args.prune_rate)
    metadata.update(pruning=pruning, status="success")
    save_policy(args.output, policy, metadata)
    print(f"Saved {args.output.resolve()}")
    print(f"Pruning {json.dumps(pruning)}")


if __name__ == "__main__":
    main()
