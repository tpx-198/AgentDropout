"""Shared command-line configuration for execution and edge training."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from ad_l.models import ModelSettings, create_model
from ad_l.roles import create_team
from ad_l.topology import EdgePolicy, TEMPORAL_MODES, TOPOLOGIES, TopologyConfig, load_policy

ROOT = Path(__file__).resolve().parents[1]


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--agents", type=int, help="Total agents including the decision-maker; default 5")
    parser.add_argument("--rounds", type=int, help="Discussion rounds; default 2")
    parser.add_argument("--topology", choices=TOPOLOGIES, help="Candidate graph; default full")
    parser.add_argument("--temporal", choices=TEMPORAL_MODES, help="Previous-round candidate graph; default upper")
    parser.add_argument("--initial-weight", type=float, help="Initial edge probability; default 0.5")
    parser.add_argument("--density", type=float, help="Random candidate graph density; default 0.5")
    parser.add_argument("--layers", type=int, help="Number of layers; default 2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--roles", nargs="+", help="One quoted role name per discussion agent")
    parser.add_argument("--deepseek", action="store_true", help="Authorize real DeepSeek API requests")
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float, help="LLM temperature")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--mock-response", default="[MOCK fixed response]\nThe answer is 5")
    parser.add_argument("--plan", action="store_true", help="Print call budget without creating model clients")


def resolve_policy(args, checkpoint: Path | None = None):
    defaults = asdict(TopologyConfig())
    names = ("agents", "rounds", "topology", "temporal", "initial_weight", "density", "layers")
    if checkpoint is not None:
        policy, metadata = load_policy(checkpoint)
        saved = asdict(policy.config)
        for name in names:
            requested = getattr(args, name)
            if requested is not None and requested != saved[name]:
                raise ValueError(f"--{name.replace('_', '-')} conflicts with the checkpoint")
        roles = args.roles if args.roles is not None else metadata.get("roles")
    else:
        values = {name: defaults[name] if getattr(args, name) is None else getattr(args, name) for name in names}
        policy = EdgePolicy(TopologyConfig(**values, seed=args.seed))
        roles = args.roles
    team = create_team(policy.config.agents, roles)
    return policy, team


def adapters_from_args(args, team):
    if args.deepseek:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
        settings = ModelSettings(backend="deepseek", model=args.model or os.environ.get("DEEPSEEK_MODEL", ""),
                                 temperature=args.temperature, max_tokens=args.max_tokens, timeout=args.timeout)
    else:
        if any(value is not None for value in (args.model, args.temperature, args.max_tokens)):
            raise ValueError("LLM generation parameters require --deepseek")
        settings = ModelSettings()
    adapter = create_model(settings, allow_real_api=args.deepseek, mock_response=args.mock_response)
    return {agent.node_id: adapter for agent in team}, settings


def print_plan(policy, team, examples=1):
    calls = examples * (policy.config.workers * policy.config.rounds + 1)
    return {"agents": [asdict(agent) for agent in team], "topology": asdict(policy.config),
            "planned_model_calls": calls}


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
