"""GSM8K 多步训练、原子检查点和可复现的数据位置恢复。"""

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch

from ad_l.evaluation import write_json
from ad_l.topology import TopologyPolicy
from ad_l.topology_training import topology_training_step


def load_training_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("schema_version") != 2 or checkpoint.get("context", {}).get("dataset", {}).get("split") != "train"
            or type(checkpoint.get("step")) is not int or checkpoint["step"] < 0):
        raise ValueError("需要 GSM8K train 划分的 v2 训练检查点；旧合成题检查点不可用于此入口")
    cursor = checkpoint.get("cursor")
    batch_size = checkpoint["context"].get("hyperparameters", {}).get("batch_size")
    if type(cursor) is not int or cursor < 0 or type(batch_size) is not int or batch_size < 1 or cursor != checkpoint["step"] * batch_size:
        raise ValueError("检查点数据位置与训练步数不一致")
    return checkpoint


def policy_from_checkpoint(checkpoint):
    state = checkpoint["model_state_dict"]
    policy = TopologyPolicy(state["spatial_masks"], state["temporal_masks"])
    policy.load_state_dict(state)
    if not all(torch.isfinite(parameter).all() for parameter in policy.parameters()):
        raise ValueError("检查点参数必须有限")
    return policy


def save_training_checkpoint(path, policy, optimizer, generator, step, cursor, context):
    temporary = path.with_suffix(".tmp")
    torch.save({"schema_version": 2, "step": step, "cursor": cursor, "context": context,
                "model_state_dict": {name: value.detach().clone() for name, value in policy.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(), "generator_state": generator.get_state()}, temporary)
    temporary.replace(path)


def train_multistep(pool, config, settings, context, output_dir, total_steps, *, checkpoint=None,
                    adapter_factory=None, allow_real_api=False):
    if not pool or any(sample.split != "train" for sample in pool):
        raise ValueError("拓扑训练只能使用 train 划分")
    if settings.backend != "mock" and not allow_real_api:
        raise ValueError("真实训练需要显式启用")
    hp = context["hyperparameters"]
    if checkpoint is not None and checkpoint["context"] != context:
        raise ValueError("续训的数据、模型、提示词、源码或训练参数与检查点不一致")
    if checkpoint and checkpoint.get("evaluation_only"):
        raise ValueError("独立剪枝产物仅供评测；当前训练入口不接收它续训")
    if context.get("settings") != asdict(settings) or context.get("config") != config.to_dict():
        raise ValueError("上下文必须记录实际模型和通信配置")
    if context.get("pool_sample_ids") != [sample.sample_id for sample in pool]:
        raise ValueError("训练样本池与上下文不一致")
    policy = policy_from_checkpoint(checkpoint) if checkpoint else TopologyPolicy(
        config.spatial_masks, config.temporal_masks, hp["initial_probability"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=hp["lr"])
    generator = torch.Generator().manual_seed(hp["seed"])
    start, cursor = 0, 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        generator.set_state(checkpoint["generator_state"])
        start, cursor = checkpoint["step"], checkpoint["cursor"]
    if type(total_steps) is not int or total_steps <= start:
        raise ValueError("steps 是目标总步数，必须大于检查点已完成步数")
    order = list(range(len(pool)))
    random.Random(hp["seed"]).shuffle(order)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "manifest.json", {"schema_version": 2, "context": context,
        "step_start": start, "target_steps": total_steps, "cursor_start": cursor,
        "sampling_order_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest()})
    save_training_checkpoint(output_dir / "checkpoint.pt", policy, optimizer, generator, start, cursor, context)
    write_json(output_dir / "summary.json", {"schema_version": 2, "status": "running", "step_start": start,
        "completed_steps": start, "target_steps": total_steps, "data_cursor": cursor,
        "backend": settings.backend, "model_calls_this_run": 0, "real_api_calls": 0})
    reports = []
    with (output_dir / "history.jsonl").open("x", encoding="utf-8") as stream:
        for step in range(start + 1, total_steps + 1):
            batch = [pool[order[(cursor + offset) % len(pool)]] for offset in range(hp["batch_size"])]
            examples = [{"sample_id": sample.sample_id, "task": sample.question, "target": sample.target,
                         "data_split": "train"} for sample in batch]
            rng_before = generator.get_state()
            report = topology_training_step(policy, optimizer, config, examples, generator=generator,
                temperature=hp["temperature"], delta=hp["delta"], settings=settings,
                adapter_factory=adapter_factory, allow_real_api=allow_real_api)
            report.update({"step": step, "data_split": "train", "backend": settings.backend,
                           "cursor_before": cursor, "sample_ids": [sample.sample_id for sample in batch]})
            if report["status"] == "success":
                cursor += hp["batch_size"]
                save_training_checkpoint(output_dir / "checkpoint.pt", policy, optimizer, generator, step, cursor, context)
            else:
                # 未完成批次不更新参数，恢复采样随机状态；续训从最近成功步重试。
                generator.set_state(rng_before)
            reports.append(report)
            stream.write(json.dumps(report, ensure_ascii=False) + "\n")
            stream.flush()
            write_json(output_dir / "training.json", report)
            calls = [call for item in reports for episode in item["episodes"] for call in episode["trace"]]
            known = {key: sum(call["usage"][key] for call in calls if call["usage"] is not None)
                     for key in ("input_tokens", "output_tokens", "total_tokens")}
            complete = bool(calls) and all(call["usage"] is not None for call in calls)
            summary = {"schema_version": 2, "status": "failed" if report["status"] != "success" else (
                "success" if step == total_steps else "running"), "step_start": start,
                "completed_steps": step if report["status"] == "success" else step - 1,
                "target_steps": total_steps, "data_cursor": cursor, "backend": settings.backend,
                "model_calls_this_run": len(calls), "real_api_calls": sum(call["real_api_attempted"] for call in calls),
                "usage": known if complete else None, "known_usage": known, "usage_complete": complete,
                "last_loss": report.get("loss"),
                "metrics_kind": "mock_diagnostic" if settings.backend == "mock" else "model_training"}
            write_json(output_dir / "summary.json", summary)
            print(f"step {step}/{total_steps}: {report['status']}，"
                  f"loss={report.get('loss')}，reward={report.get('rewards')}")
            if report["status"] != "success":
                return summary
    return summary
