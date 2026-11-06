"""PyTorch 拓扑参数与原共享／逐轮独立分支的采样。"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


from ad_l.communication import execution_topology


@dataclass(frozen=True)
class SampledRound:
    round_index: int
    spatial_edges: tuple
    temporal_edges: tuple
    log_prob: torch.Tensor
    decisions: tuple
    node_ids: tuple = ("A", "B")

    def for_execution(self):
        return execution_topology(self.spatial_edges, self.temporal_edges, self.node_ids)

    def for_report(self):
        return {"round": self.round_index + 1, **self.for_execution(),
                "log_prob": self.log_prob.detach().item(), "decisions": list(self.decisions)}


class TopologyPolicy(nn.Module):
    """diff=True 时两轮空间参数、一组跨轮时间参数各自独立。"""

    def __init__(self, spatial_masks, temporal_masks, initial_probability=0.5, *, diff=False, node_ids=("A", "B")):
        super().__init__()
        if type(diff) is not bool:
            raise ValueError("diff 必须是布尔值")
        self.diff = diff
        self.node_ids = tuple(node_ids)
        size = len(self.node_ids)
        if size < 2 or len(set(self.node_ids)) != size:
            raise ValueError("拓扑需要至少两个不同的节点")
        matrix_shape = (size, size)
        if not math.isfinite(initial_probability) or not 0 < initial_probability < 1:
            raise ValueError("initial_probability 必须严格位于 0 和 1 之间")
        for name, masks in [("spatial_masks", spatial_masks), ("temporal_masks", temporal_masks)]:
            tensor = torch.as_tensor(masks, dtype=torch.float32)
            shape = ((2 if name == "spatial_masks" else 1), *matrix_shape) if diff else matrix_shape
            if diff and tensor.shape == matrix_shape:
                tensor = tensor.expand(shape)
            if tensor.shape != shape or not torch.all((tensor == 0) | (tensor == 1)):
                raise ValueError(f"{name} 必须是形状 {shape} 的 0/1 矩阵")
            self.register_buffer(name, tensor.clone())
        initial_logit = math.log(initial_probability / (1 - initial_probability))
        self.spatial_logits = nn.Parameter(torch.full_like(self.spatial_masks, initial_logit))
        self.temporal_logits = nn.Parameter(torch.full_like(self.temporal_masks, initial_logit))

    def sample_rounds(self, rounds=2, temperature=1.0, *, generator=None):
        if type(rounds) is not int or rounds < 1:
            raise ValueError("rounds 必须是正整数")
        if self.diff and rounds != 2:
            raise ValueError("逐轮独立模式当前只支持两轮")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature 必须是有限正数")
        if not all(torch.isfinite(parameter).all() for parameter in self.parameters()):
            raise ValueError("拓扑参数必须有限")
        nodes = self.node_ids
        result = []
        for round_index in range(rounds):
            spatial, temporal, decisions = [], [], []
            # 零锚点让全空 mask 的 backward 也合法，不增加任何采样概率项。
            terms = [(self.spatial_logits.sum() + self.temporal_logits.sum()) * 0]

            def reaches(start, target):
                pending, seen = [start], set()
                while pending:
                    node = pending.pop()
                    if node == target:
                        return True
                    if node not in seen:
                        seen.add(node)
                        pending.extend(dst for src, dst in spatial if src == node)
                return False

            for kind, logits, mask, edges in [
                ("spatial", self.spatial_logits, self.spatial_masks, spatial),
                ("temporal", self.temporal_logits, self.temporal_masks, temporal),
            ]:
                if kind == "temporal" and round_index == 0:
                    continue
                if self.diff:
                    index = round_index if kind == "spatial" else round_index - 1
                    logits, mask = logits[index], mask[index]
                for i, source in enumerate(nodes):
                    for j, target in enumerate(nodes):
                        if mask[i, j].item() == 0 or (kind == "spatial" and reaches(target, source)):
                            continue
                        scaled = logits[i, j] / temperature
                        probability = torch.sigmoid(scaled)
                        selected = bool((torch.rand(1, generator=generator, device=logits.device) < probability).item())
                        if selected:
                            edges.append((source, target))
                        # 稳定计算 log(p) / log(1-p)，不截断梯度。
                        terms.append(F.logsigmoid(scaled if selected else -scaled))
                        decisions.append({"kind": kind, "source": source, "target": target,
                                          "selected": selected, "probability": probability.detach().item()})
            result.append(SampledRound(round_index, tuple(spatial), tuple(temporal),
                                       torch.stack(terms).sum(), tuple(decisions), nodes))
        return result

    def regularization_loss(self, delta=0.1, *, reference_masks=None):
        """原 GSM8K 主训练循环的核范数与 Frobenius 约束。"""
        if not math.isfinite(delta) or delta < 0:
            raise ValueError("delta 必须是有限非负数")
        masks = reference_masks or (self.spatial_masks, self.temporal_masks)
        result = []
        for logits, mask in zip((self.spatial_logits, self.temporal_logits), masks):
            mask = torch.as_tensor(mask, dtype=logits.dtype, device=logits.device)
            # 原 diff=True：分别求各轮均值，再对 Frobenius 均值施加一次阈值。
            nuclear = torch.linalg.svdvals(logits).sum(dim=-1).mean()
            frobenius = torch.linalg.matrix_norm(mask - logits, ord="fro").mean()
            result.append(nuclear + F.relu(frobenius - delta))
        return sum(result)

    def fixed_round_topologies(self, config):
        """逐轮复现原非优化建边规则；第一轮没有跨轮输入。"""
        from dataclasses import replace
        from ad_l.fixed_graph import resolve_fixed_topology
        if self.node_ids != config.node_ids:
            raise ValueError("策略与配置的节点顺序不一致")
        result = []
        for round_index in range(2):
            spatial = self.spatial_masks[round_index] if self.diff else self.spatial_masks
            temporal = self.temporal_masks[0] if self.diff else self.temporal_masks
            topology = resolve_fixed_topology(replace(config,
                spatial_masks=tuple(tuple(int(v) for v in row) for row in spatial.tolist()),
                temporal_masks=tuple(tuple(int(v) for v in row) for row in temporal.tolist())))
            if round_index == 0:
                topology["temporal_edges"] = []
            result.append(topology)
        return result
