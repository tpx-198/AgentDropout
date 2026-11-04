"""PyTorch 拓扑参数与原 diff=False 优化分支的逐轮采样。"""

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

    def for_execution(self):
        return execution_topology(self.spatial_edges, self.temporal_edges)

    def for_report(self):
        return {"round": self.round_index + 1, **self.for_execution(),
                "log_prob": self.log_prob.detach().item(), "decisions": list(self.decisions)}


class TopologyPolicy(nn.Module):
    """两节点共享轮次参数；mask=0 不采样，空间边按原顺序逐条拒绝环路。"""

    def __init__(self, spatial_masks, temporal_masks, initial_probability=0.5):
        super().__init__()
        if not math.isfinite(initial_probability) or not 0 < initial_probability < 1:
            raise ValueError("initial_probability 必须严格位于 0 和 1 之间")
        for name, masks in [("spatial_masks", spatial_masks), ("temporal_masks", temporal_masks)]:
            tensor = torch.as_tensor(masks, dtype=torch.float32)
            if tensor.shape != (2, 2) or not torch.all((tensor == 0) | (tensor == 1)):
                raise ValueError("mask 必须是 2×2 的 0/1 矩阵")
            self.register_buffer(name, tensor.clone())
        initial_logit = math.log(initial_probability / (1 - initial_probability))
        self.spatial_logits = nn.Parameter(torch.full((2, 2), initial_logit))
        self.temporal_logits = nn.Parameter(torch.full((2, 2), initial_logit))

    def sample_rounds(self, rounds=2, temperature=1.0, *, generator=None):
        if type(rounds) is not int or rounds < 1:
            raise ValueError("rounds 必须是正整数")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature 必须是有限正数")
        if not all(torch.isfinite(parameter).all() for parameter in self.parameters()):
            raise ValueError("拓扑参数必须有限")
        nodes = ("A", "B")
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
                                       torch.stack(terms).sum(), tuple(decisions)))
        return result

    def regularization_loss(self, delta=0.1):
        """原 GSM8K 主训练循环的核范数与 Frobenius 约束。"""
        if not math.isfinite(delta) or delta < 0:
            raise ValueError("delta 必须是有限非负数")
        return sum(torch.linalg.svdvals(logits).sum()
                   + F.relu(torch.linalg.matrix_norm(mask - logits, ord="fro") - delta)
                   for logits, mask in [(self.spatial_logits, self.spatial_masks),
                                        (self.temporal_logits, self.temporal_masks)])
