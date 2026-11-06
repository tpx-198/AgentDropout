"""GSM8K 多步训练、原子检查点和可复现的数据位置恢复。"""

import hashlib
import json
import random
import copy
import math
from dataclasses import asdict
from pathlib import Path

import torch

from ad_l.evaluation import write_json
from ad_l.fixed_graph import OfflineFixedGraphConfig
from ad_l.node_pruning import node_pruned_masks
from ad_l.topology import TopologyPolicy
from ad_l.topology_training import topology_training_step


def pruning_schedule(context):
    hp = context["hyperparameters"]
    every, rate, until = hp.get("prune_every", 0), hp.get("prune_rate", 0.25), hp.get("prune_until")
    if (type(every) is not int or every < 0 or type(rate) not in (int, float)
            or not math.isfinite(rate) or not 0 <= rate <= 1):
        raise ValueError("prune-every 必须是非负整数，prune-rate 必须在 0 到 1 之间")
    if until is not None and (type(until) is not int or until < 1):
        raise ValueError("prune-until 必须是正整数或省略")
    return {"every": every, "rate": rate, "until": until}


def pruning_due(step, schedule):
    return (schedule["every"] > 0 and step % schedule["every"] == 0
            and (schedule["until"] is None or step <= schedule["until"]))


def validate_pruning_state(checkpoint):
    schedule = pruning_schedule(checkpoint["context"])
    state = checkpoint.get("pruning_state")
    if state is None:
        if schedule["every"]:
            raise ValueError("周期剪枝检查点缺少剪枝进度")
        return {"schedule": schedule, "events": []}
    if state.get("schedule") != schedule or not isinstance(state.get("events"), list):
        raise ValueError("剪枝进度与计划不一致")
    bound = min(checkpoint["step"], schedule["until"]) if schedule["until"] is not None else checkpoint["step"]
    count = bound // schedule["every"] if schedule["every"] else 0
    if len(state["events"]) != count or any(not isinstance(event, dict) or event.get("step") != (i + 1) * schedule["every"]
                                            for i, event in enumerate(state["events"])):
        raise ValueError("剪枝事件缺失、重复或超出已完成训练步数")
    return copy.deepcopy(state)


def load_training_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("schema_version") != 2 or checkpoint.get("context", {}).get("dataset", {}).get("split") != "train"
            or type(checkpoint.get("step")) is not int or checkpoint["step"] < 0):
        raise ValueError("需要 GSM8K train 划分的 v2 训练检查点；旧合成题检查点不可用于此入口")
    cursor = checkpoint.get("cursor")
    batch_size = checkpoint["context"].get("hyperparameters", {}).get("batch_size")
    if type(cursor) is not int or cursor < 0 or type(batch_size) is not int or batch_size < 1 or cursor != checkpoint["step"] * batch_size:
        raise ValueError("检查点数据位置与训练步数不一致")
    validate_pruning_state(checkpoint)
    return checkpoint


def node_pruning_masks(context, config):
    pruning = context["node_pruning"]
    if context.get("diff") is not True:
        raise ValueError("节点剪枝只支持逐轮独立边拓扑")
    spatial, temporal = node_pruned_masks(config, pruning["skip_plan"])
    if pruning["spatial_masks"] != spatial or pruning["temporal_masks"] != temporal:
        raise ValueError("初始剪枝掩码与节点跳过计划不一致")
    return spatial, temporal


def policy_from_checkpoint(checkpoint):
    state = checkpoint["model_state_dict"]
    context = checkpoint.get("context", {})
    diff = context.get("diff", False)
    if type(diff) is not bool:
        raise ValueError("检查点 diff 必须是布尔值")
    config = OfflineFixedGraphConfig.from_dict(context["config"]) if "config" in context else None
    node_ids = config.node_ids if config is not None else ("A", "B")
    matrix = (len(node_ids), len(node_ids))
    shapes = {"spatial_masks": (2, *matrix) if diff else matrix,
              "temporal_masks": (1, *matrix) if diff else matrix}
    if any(state[name].shape != shape or state[name.replace("masks", "logits")].shape != shape
           for name, shape in shapes.items()):
        raise ValueError("检查点参数形状与 diff 模式不一致")
    policy = TopologyPolicy(state["spatial_masks"], state["temporal_masks"], diff=diff, node_ids=node_ids)
    if "node_pruning" in context:
        masks = node_pruning_masks(context, config)
        for kind, initial in zip(("spatial", "temporal"), masks):
            if (getattr(policy, kind + "_masks") > torch.tensor(initial)).any():
                raise ValueError("检查点重新启用了已剪除的候选边")
    try:
        policy.load_state_dict(state)
    except RuntimeError as error:
        raise ValueError("检查点拓扑参数不完整") from error
    if not all(torch.isfinite(parameter).all() for parameter in policy.parameters()):
        raise ValueError("检查点参数必须有限")
    return policy


def save_training_checkpoint(path, policy, optimizer, generator, step, cursor, context, *, pruning_state=None):
    temporary = path.with_suffix(".tmp")
    torch.save({"schema_version": 2, "step": step, "cursor": cursor, "context": context,
                "model_state_dict": {name: value.detach().clone() for name, value in policy.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(), "generator_state": generator.get_state(),
                "pruning_state": copy.deepcopy(pruning_state) if pruning_state is not None else {
                    "schedule": pruning_schedule(context), "events": []}}, temporary)
    temporary.replace(path)


def train_multistep(pool, config, settings, context, output_dir, total_steps, *, checkpoint=None,
                    adapter_factory=None, allow_real_api=False):
    if not pool or any(sample.split != "train" for sample in pool):
        raise ValueError("拓扑训练只能使用 train 划分")
    if settings.backend != "mock" and not allow_real_api:
        raise ValueError("真实训练需要显式启用")
    hp = context["hyperparameters"]
    schedule = pruning_schedule(context)
    if checkpoint is not None and checkpoint["context"] != context:
        raise ValueError("续训的数据、模型、提示词、源码或训练参数与检查点不一致")
    if checkpoint and checkpoint.get("evaluation_only"):
        raise ValueError("独立剪枝产物仅供评测；当前训练入口不接收它续训")
    if context.get("settings") != asdict(settings) or context.get("config") != config.to_dict():
        raise ValueError("上下文必须记录实际模型和通信配置")
    if context.get("pool_sample_ids") != [sample.sample_id for sample in pool]:
        raise ValueError("训练样本池与上下文不一致")
    skip_nodes = (None, None)
    masks = config.spatial_masks, config.temporal_masks
    if "node_pruning" in context:
        masks = node_pruning_masks(context, config)
        skip_nodes = tuple(context["node_pruning"]["skip_plan"])
    policy = policy_from_checkpoint(checkpoint) if checkpoint else TopologyPolicy(
        *masks, hp["initial_probability"], diff=context.get("diff", False), node_ids=config.node_ids)
    optimizer = torch.optim.Adam(policy.parameters(), lr=hp["lr"])
    generator = torch.Generator().manual_seed(hp["seed"])
    pruning_state = {"schedule": schedule, "events": []}
    start, cursor = 0, 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        generator.set_state(checkpoint["generator_state"])
        start, cursor = checkpoint["step"], checkpoint["cursor"]
        pruning_state = validate_pruning_state(checkpoint)
    if type(total_steps) is not int or total_steps <= start:
        raise ValueError("steps 是目标总步数，必须大于检查点已完成步数")
    order = list(range(len(pool)))
    random.Random(hp["seed"]).shuffle(order)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "manifest.json", {"schema_version": 2, "context": context,
        "step_start": start, "target_steps": total_steps, "cursor_start": cursor,
        "pruning_schedule": schedule,
        "sampling_order_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest()})
    save_training_checkpoint(output_dir / "checkpoint.pt", policy, optimizer, generator, start, cursor, context,
                             pruning_state=pruning_state)
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
                adapter_factory=adapter_factory, allow_real_api=allow_real_api, skip_nodes=skip_nodes)
            report.update({"step": step, "data_split": "train", "backend": settings.backend,
                           "cursor_before": cursor, "sample_ids": [sample.sample_id for sample in batch]})
            report["pruning_event"] = None
            if report["status"] == "success":
                cursor += hp["batch_size"]
                if pruning_due(step, schedule):
                    # 延迟导入，复用已核对的原剪枝规则；剪枝不重置 Adam，也不抽随机数。
                    from ad_l.pruning import prune_edges, fixed_execution
                    effective_before = fixed_execution(policy, config)
                    event = {"step": step, "phase": "after_optimizer_step", **prune_edges(policy, schedule["rate"]),
                             "effective_before": effective_before, "effective_after": fixed_execution(policy, config)}
                    pruning_state["events"].append(event)
                    report["pruning_event"] = event
                save_training_checkpoint(output_dir / "checkpoint.pt", policy, optimizer, generator, step, cursor, context,
                                         pruning_state=pruning_state)
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
                "pruning_events_total": len(pruning_state["events"]),
                "pruning_events_this_run": sum(event["step"] > start for event in pruning_state["events"]),
                "remaining_candidates": {kind: int(getattr(policy, kind + "_masks").sum().item())
                                         for kind in ("spatial", "temporal")},
                "metrics_kind": "mock_diagnostic" if settings.backend == "mock" else "model_training"}
            write_json(output_dir / "summary.json", summary)
            print(f"step {step}/{total_steps}: {report['status']}，"
                  f"loss={report.get('loss')}，reward={report.get('rewards')}")
            if report["pruning_event"] is not None:
                changes = report["pruning_event"]["changes"]
                print("  周期边剪枝：" + "；".join(
                    f"{kind} {item['active_before']} → {item['active_after']}" for kind, item in changes.items()))
            if report["status"] != "success":
                return summary
    return summary
