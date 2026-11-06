"""GSM8K 节点选择训练与逐轮节点剪枝。"""

import hashlib
import json
from time import perf_counter

import torch
from langsmith import tracing_context
from torch.nn import functional as F

from ad_l.evaluation import score_answer
from ad_l.fixed_graph import build_fixed_graph
from ad_l.topology import TopologyPolicy

def sample_skips(spatial_logits, topologies, generator, *, node_ids=("A", "B")):
    """按原 skip 分支从 incident logit 和的 softmax 抽取节点，保留原替代损失。"""
    plan, probabilities = [], []
    surrogate = spatial_logits.sum() * 0
    for logits, topology in zip(spatial_logits, topologies):
        incident = [[] for _ in node_ids]
        for source, target in topology["spatial_edges"]:
            i, j = node_ids.index(source), node_ids.index(target)
            incident[i].append(logits[i, j])
            incident[j].append(logits[i, j])
        # 原采样分布与梯度断开；反传使用下面的边级替代项。
        scores = torch.stack([sum(edges, logits.new_zeros(())) for edges in incident])
        distribution = scores.detach().softmax(dim=0)
        selected = torch.multinomial(distribution, 1, generator=generator).item()
        plan.append(node_ids[selected])
        probabilities.append(distribution.tolist())
        for i, edges in enumerate(incident):
            for edge_logit in edges:
                surrogate = surrogate + (4 * F.logsigmoid(-edge_logit) if i == selected else F.logsigmoid(edge_logit))
    return tuple(plan), probabilities, surrogate


def train_selector_step(selector, optimizer, config, samples, *, generator, adapters=None, allow_real_api=False):
    if not selector.diff or not samples or any(sample.split != "train" for sample in samples):
        raise ValueError("节点选择训练需要逐轮独立参数和非空 GSM8K train 批次")
    episodes, losses = [], []
    before = selector.spatial_logits.detach().tolist()
    topologies = selector.fixed_round_topologies(config)
    with tracing_context(enabled=False):
        for sample in samples:
            started = perf_counter()
            plan, probabilities, surrogate = sample_skips(selector.spatial_logits, topologies, generator,
                                                           node_ids=config.node_ids)
            graph = build_fixed_graph(config, adapters, allow_real_api=allow_real_api,
                                      round_topologies=topologies, skip_nodes=plan)
            state = graph.invoke({"task": sample.question})
            score = score_answer(state["answer"], sample.target, state["status"])
            episodes.append({"sample_id": sample.sample_id, "question": sample.question, "target": sample.target,
                             "baseline": "selector", "skip_plan": plan, "skip_probabilities": probabilities,
                             "surrogate_log_prob": surrogate.detach().item(), **score,
                             **{key: state[key] for key in ("status", "answer", "trace", "skipped_nodes",
                                                            "round_outputs", "setup_error", "model_calls",
                                                            "real_api_calls")},
                             "elapsed_seconds": perf_counter() - started})
            if state["status"] != "success":
                return {"status": "failed", "episodes": episodes}
            losses.append(-surrogate * float(score["correct"]))
    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = selector.spatial_logits.grad
    if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
        optimizer.zero_grad(set_to_none=True)
        return {"status": "failed", "error": "nonfinite_loss_or_gradient", "episodes": episodes}
    optimizer.step()
    after = selector.spatial_logits.detach().tolist()
    return {"status": "success", "loss": loss.detach().item(), "episodes": episodes,
            "spatial_logits_before": before, "spatial_logits_after": after,
            "spatial_gradient": gradient.detach().tolist(), "parameters_changed": before != after}


def fixed_degrees(config):
    mask = torch.tensor(config.spatial_masks)
    degrees = mask.sum(dim=0) + mask.sum(dim=1)
    if (degrees == 0).any():
        raise ValueError("节点剪枝要求初始空间矩阵中每个智能体的入度与出度之和大于零")
    return degrees


def node_pruned_masks(config, plan):
    if (not isinstance(plan, (list, tuple)) or len(plan) != 2
            or any(node not in config.node_ids for node in plan)):
        raise ValueError("节点剪枝计划必须指定两轮各跳过一个已配置节点")
    spatial = torch.tensor(config.spatial_masks).repeat(2, 1, 1)
    temporal = torch.tensor(config.temporal_masks).unsqueeze(0)
    for round_index, node in enumerate(plan):
        index = config.node_ids.index(node)
        spatial[round_index, index, :] = 0
        spatial[round_index, :, index] = 0
        if round_index > 0:
            temporal[round_index - 1, :, index] = 0
        if round_index + 1 < len(plan):
            temporal[round_index, index, :] = 0
    return spatial.tolist(), temporal.tolist()


def load_node_pruning(path, config):
    raw = path.read_bytes()
    report = json.loads(raw)
    if not isinstance(report, dict):
        raise ValueError("节点剪枝报告必须为 JSON 对象")
    source_config = report.get("source_config")
    if source_config is None:
        # 兼容此前的 nodes.py 产物，初始配置保存在同目录的实验清单。
        manifest = json.loads(path.with_name("manifest.json").read_bytes())
        source_config = manifest.get("config") if isinstance(manifest, dict) else None
    if source_config != config.to_dict() or report.get("node_order") != list(config.node_ids):
        raise ValueError("节点剪枝报告的初始配置或节点顺序与本次训练不一致")
    plan = report.get("skip_plan")
    spatial, temporal = node_pruned_masks(config, plan)
    if report.get("spatial_masks") != spatial or report.get("temporal_masks") != temporal:
        raise ValueError("节点剪枝报告的掩码与跳过计划不一致")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
            "skip_plan": plan, "spatial_masks": spatial, "temporal_masks": temporal}


@torch.no_grad()
def prune_nodes(selector, config):
    """每轮删除一个节点的空间边、上一轮来边与下一轮出边。"""
    logits = selector.spatial_logits
    if not selector.diff or selector.node_ids != config.node_ids or not torch.isfinite(logits).all():
        raise ValueError("节点剪枝需要有限的逐轮选择权重")
    scores = (logits.sum(dim=1) + logits.sum(dim=2)) / fixed_degrees(config)
    indices = scores.argmin(dim=1).tolist()
    before = TopologyPolicy(config.spatial_masks, config.temporal_masks, diff=True, node_ids=config.node_ids)
    plan = tuple(config.node_ids[index] for index in indices)
    spatial, temporal = node_pruned_masks(config, plan)
    after = TopologyPolicy(spatial, temporal, diff=True, node_ids=config.node_ids)
    report = {"skip_plan": plan, "node_order": config.node_ids, "scores": scores.tolist(),
              "source_config": config.to_dict(),
              "selector_spatial_logits": logits.tolist(),
              "selector_temporal_logits": selector.temporal_logits.tolist(),
              "spatial_masks": after.spatial_masks.tolist(), "temporal_masks": after.temporal_masks.tolist(),
              "effective_before": before.fixed_round_topologies(config),
              "effective_after": after.fixed_round_topologies(config)}
    return before, after, plan, report
