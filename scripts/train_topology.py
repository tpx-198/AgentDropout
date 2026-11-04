"""只做一次离线拓扑更新，不读取密钥，不发送模型请求。"""

import argparse
import hashlib
import importlib.metadata as metadata
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from ad_l.evaluation import SCORER, write_json
from ad_l.fixed_graph import load_fixed_graph_config
from ad_l.topology import TopologyPolicy
from ad_l.topology_training import offline_training_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--initial-probability", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fixed_graph_offline.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr 必须是有限正数")
    try:
        config = load_fixed_graph_config(args.config)
        policy = TopologyPolicy(config.spatial_masks, config.temporal_masks, args.initial_probability)
        if not math.isfinite(args.temperature) or args.temperature <= 0 or not math.isfinite(args.delta) or args.delta < 0:
            raise ValueError("temperature 必须是有限正数；delta 必须是有限非负数")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    run_id = uuid4().hex
    output = args.output_dir or ROOT / "results/topology" / run_id
    if output.exists():
        parser.error("output-dir 已存在，请使用新目录")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    examples = [{"sample_id": "synthetic_train:apples:0", "task": "Anna has 3 apples and buys 2 more. How many apples does she have?", "target": "5"}]
    report = offline_training_step(policy, optimizer, config, examples,
                                   generator=generator, temperature=args.temperature, delta=args.delta)
    source_files = [ROOT / f"src/ad_l/{name}.py" for name in ("communication", "topology", "topology_training", "fixed_graph", "models", "prompts", "evaluation")]
    report.update({"schema_version": 1, "run_id": run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "backend": "mock", "data_split": "synthetic_train", "training_steps": 1,
                   "seed": args.seed, "provider_seed": None, "optimizer": "Adam", "lr": args.lr,
                   "initial_probability": args.initial_probability, "temperature": args.temperature,
                   "delta": args.delta, "scorer": SCORER, "diff": False,
                   "spatial_masks": policy.spatial_masks.tolist(), "temporal_masks": policy.temporal_masks.tolist(),
                   "config": config.to_dict(), "python": platform.python_version(),
                   "versions": {name: metadata.version(name) for name in ("torch", "langgraph", "langchain-core")},
                   "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
                   "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()})
    output.mkdir(parents=True)
    write_json(output / "training.json", report)
    # 仅保存已脱离计算图的参数、优化器状态和 RNG 状态；不保存采样 log_prob。
    torch.save({"schema_version": 1, "step": 1,
                "model_state_dict": {name: value.detach().clone() for name, value in policy.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(), "generator_state": generator.get_state(),
                "seed": args.seed, "temperature": args.temperature, "delta": args.delta,
                "initial_probability": args.initial_probability, "lr": args.lr}, output / "checkpoint.pt")
    print(f"离线训练更新：1；奖励：{report['rewards']}；损失：{report['loss']:.6f}")
    print(f"log_prob 保留计算图：{report['sampling_log_prob_requires_grad']}；参数已更新：{report['parameters_changed']}")
    print(f"模型调用：{report['model_calls']} 次模拟；真实请求：0")
    print(f"训练记录：{output.resolve()}")


if __name__ == "__main__":
    main()
