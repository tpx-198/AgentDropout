"""原 diff=False 边剪枝规则、来源校验与配对指标。"""

import copy
import math
from dataclasses import replace

import torch

from ad_l.fixed_graph import resolve_fixed_topology
from ad_l.training import policy_from_checkpoint

RULE = "agentdropout_update_masks_v1"


def fixed_execution(policy, config):
    return resolve_fixed_topology(replace(config,
        spatial_masks=tuple(tuple(int(value) for value in row) for row in policy.spatial_masks.tolist()),
        temporal_masks=tuple(tuple(int(value) for value in row) for row in policy.temporal_masks.tolist())))


@torch.no_grad()
def prune_edges(policy, rate=0.25):
    if not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("pruning rate 必须在 0 到 1 之间")
    records = {}
    for kind in ("spatial", "temporal"):
        mask = getattr(policy, kind + "_masks")
        logits = getattr(policy, kind + "_logits")
        if not torch.isfinite(logits).all():
            raise ValueError("不能按非有限权重剪枝")
        before = mask.clone()
        flat_mask = mask.view(-1)
        active = (flat_mask > 0).sum()
        disabled = (flat_mask == 0).sum()
        requested = max(int(torch.round(active * rate).item()), 1)
        ranked = logits.detach().flatten().clone()
        ranked[flat_mask == 0] = ranked.min() - 1.0
        indices = torch.argsort(ranked)[:requested + int(disabled.item())]
        flat_mask[indices] = 0
        removed = []
        for index in range(flat_mask.numel()):
            i, j = divmod(index, 2)
            if before[i, j] == 1 and mask[i, j] == 0:
                removed.append({"source": ("A", "B")[i], "target": ("A", "B")[j],
                                "logit": logits[i, j].item()})
        count = int(active.item())
        records[kind] = {"before_mask": before.tolist(), "after_mask": mask.tolist(),
                         "active_before": count, "active_after": int(mask.sum().item()),
                         "requested_remove": requested, "actual_removed": len(removed),
                         "actual_rate": len(removed) / count if count else None, "removed_edges": removed}
    return {"rule": RULE, "rate": rate, "changes": records}


def prune_checkpoint(checkpoint, config, rate, *, parent_sha256):
    if checkpoint["step"] < 1:
        raise ValueError("检查点尚未完成训练")
    before = policy_from_checkpoint(checkpoint).eval().requires_grad_(False)
    after = policy_from_checkpoint(checkpoint).eval().requires_grad_(False)
    report = prune_edges(after, rate)
    report.update({"parent_sha256": parent_sha256, "training_step": checkpoint["step"],
                   "effective_before": fixed_execution(before, config),
                   "effective_after": fixed_execution(after, config)})
    for kind in ("spatial", "temporal"):
        key = kind + "_edges"
        previous = {tuple(edge) for edge in report["effective_before"][key]}
        following = {tuple(edge) for edge in report["effective_after"][key]}
        report["changes"][kind]["effective_removed"] = [list(edge) for edge in sorted(previous - following)]
        report["changes"][kind]["effective_added"] = [list(edge) for edge in sorted(following - previous)]
    pruned = copy.deepcopy(checkpoint)
    pruned["model_state_dict"] = {name: value.detach().clone() for name, value in after.state_dict().items()}
    report["previous_pruning"] = copy.deepcopy(checkpoint.get("pruning"))
    pruned["pruning"] = report
    # 独立剪枝产物只用于评测，避免续训时丢失剪枝来源和计划语义。
    pruned["evaluation_only"] = True
    return pruned, report, before, after


def verify_pair(before_checkpoint, after_checkpoint, before_sha256):
    if after_checkpoint.get("pruning", {}).get("parent_sha256") != before_sha256:
        raise ValueError("剪枝后检查点来源与剪枝前文件不匹配")
    if before_checkpoint["context"] != after_checkpoint["context"] or before_checkpoint["step"] != after_checkpoint["step"]:
        raise ValueError("配对检查点的训练来源或训练步数不一致")
    for kind in ("spatial", "temporal"):
        logits = kind + "_logits"
        masks = kind + "_masks"
        if not torch.equal(before_checkpoint["model_state_dict"][logits], after_checkpoint["model_state_dict"][logits]):
            raise ValueError("剪枝前后 logit 必须保持一致")
        if not torch.all(after_checkpoint["model_state_dict"][masks] <= before_checkpoint["model_state_dict"][masks]):
            raise ValueError("剪枝不能重新启用已禁用候选边")


def compare_pairs(rows, selected_count, backend):
    pairs = {}
    for row in rows:
        pairs.setdefault(row["sample_id"], {})[row["baseline"]] = row
    completed = [pair for pair in pairs.values() if "before" in pair and "after" in pair]
    successful = [pair for pair in completed if all(row["status"] == "success" for row in pair.values())]
    token_totals = {"before": 0, "after": 0}
    complete_usage = bool(successful)
    for pair in successful:
        for name in token_totals:
            trace = pair[name]["trace"]
            if not trace or any(call["usage"] is None for call in trace):
                complete_usage = False
            token_totals[name] += sum(call["usage"]["total_tokens"] for call in trace if call["usage"] is not None)
    n = len(completed)
    correct_delta = sum(int(pair["after"]["correct"]) - int(pair["before"]["correct"]) for pair in completed)
    return {"inference_mode": "original-fixed", "order": "before_then_after",
            "selected_pairs": selected_count, "paired_attempted": n, "missing_pairs": selected_count - n,
            "paired_successful": len(successful), "efficiency_pairs": len(successful),
            "accuracy_denominator": "paired_attempted_including_failures",
            "accuracy_delta_after_minus_before": correct_delta / n if n and backend != "mock" else None,
            "mock_diagnostic_correct_delta": correct_delta if backend == "mock" else None,
            "total_tokens_delta_after_minus_before": token_totals["after"] - token_totals["before"] if complete_usage else None,
            "token_saving_fraction": 1 - token_totals["after"] / token_totals["before"]
                if complete_usage and token_totals["before"] > 0 else None,
            "model_calls_delta_after_minus_before": sum(pair["after"]["model_calls"] - pair["before"]["model_calls"] for pair in successful) if successful else None,
            "mean_seconds_delta_after_minus_before": sum(pair["after"]["elapsed_seconds"] - pair["before"]["elapsed_seconds"] for pair in successful) / len(successful) if successful else None,
            "usage_complete_for_pairs": complete_usage}
