"""Candidate graphs, differentiable Bernoulli sampling, pruning and persistence."""

import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

TOPOLOGIES = ("full", "random", "layered", "chain", "star", "debate")
TEMPORAL_MODES = ("upper", "full", "self", "none")


@dataclass(frozen=True)
class TopologyConfig:
    agents: int = 5
    rounds: int = 2
    topology: str = "full"
    temporal: str = "upper"
    initial_weight: float = 0.5
    density: float = 0.5
    layers: int = 2
    seed: int = 0

    def __post_init__(self):
        if type(self.agents) is not int or self.agents < 2:
            raise ValueError("agents must be at least 2, including the decision-maker")
        if type(self.rounds) is not int or self.rounds < 1:
            raise ValueError("rounds must be a positive integer")
        if self.topology not in TOPOLOGIES or self.temporal not in TEMPORAL_MODES:
            raise ValueError("Unknown spatial or temporal topology")
        if not 0 < self.initial_weight < 1 or not 0 <= self.density <= 1:
            raise ValueError("initial_weight must be in (0, 1), density in [0, 1]")
        if type(self.layers) is not int or self.layers < 1:
            raise ValueError("layers must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")

    @property
    def workers(self):
        return self.agents - 1


def candidate_masks(config: TopologyConfig):
    size = config.workers
    rng = random.Random(config.seed)
    layers = min(size, config.layers)
    assignment = [min(i * layers // size, layers - 1) for i in range(size)]
    spatial = torch.zeros(size, size, dtype=torch.bool)
    for source in range(size):
        for target in range(size):
            if source == target:
                continue
            connected = {
                "full": True,
                "layered": assignment[target] == assignment[source] + 1,
                "chain": target == source + 1,
                "star": source == 0,
                "debate": False,
            }
            spatial[source, target] = rng.random() < config.density if config.topology == "random" else connected[config.topology]
    temporal = torch.zeros(size, size, dtype=torch.bool)
    for source in range(size):
        for target in range(size):
            temporal[source, target] = {
                "upper": source < target, "full": True,
                "self": source == target, "none": False,
            }[config.temporal]
    return spatial.repeat(config.rounds, 1, 1), temporal.repeat(config.rounds - 1, 1, 1)


def topological_order(size: int, edges) -> list[int]:
    indegree = [0] * size
    successors = [[] for _ in range(size)]
    seen = set()
    for edge in edges:
        if (not isinstance(edge, (list, tuple)) or len(edge) != 2
                or any(type(node) is not int or not 0 <= node < size for node in edge)):
            raise ValueError("Invalid spatial edge")
        source, target = edge
        if source == target or (source, target) in seen:
            raise ValueError("Duplicate or self spatial edge")
        seen.add((source, target))
        indegree[target] += 1
        successors[source].append(target)
    pending = [i for i, degree in enumerate(indegree) if degree == 0]
    order = []
    for source in pending:
        order.append(source)
        for target in successors[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                pending.append(target)
    if len(order) != size:
        raise ValueError("Spatial communication must be a DAG")
    return order


def _reaches(edges, start, target):
    pending, visited = [start], set()
    while pending:
        node = pending.pop()
        if node == target:
            return True
        if node not in visited:
            visited.add(node)
            pending.extend(destination for source, destination in edges if source == node)
    return False


@dataclass
class SampledTopology:
    execution: dict
    log_prob: torch.Tensor


class EdgePolicy(nn.Module):
    """Each round has independent edge probabilities. Final aggregation edges also train."""

    def __init__(self, config: TopologyConfig):
        super().__init__()
        self.config = config
        spatial, temporal = candidate_masks(config)
        initial_logit = math.log(config.initial_weight / (1 - config.initial_weight))
        for kind, mask in (("spatial", spatial), ("temporal", temporal),
                           ("summary", torch.ones(config.workers, dtype=torch.bool))):
            self.register_buffer(f"{kind}_mask", mask)
            self.register_parameter(f"{kind}_logits", nn.Parameter(torch.full(mask.shape, initial_logit)))

    def weights(self):
        return {kind: torch.sigmoid(getattr(self, f"{kind}_logits")) * getattr(self, f"{kind}_mask")
                for kind in ("spatial", "temporal", "summary")}

    def sample(self, generator: torch.Generator, *, mode="sampled", threshold=0.5, temperature=1.0):
        if mode not in {"sampled", "fixed", "threshold"}:
            raise ValueError("mode must be sampled, fixed or threshold")
        if not 0 <= threshold <= 1 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid threshold or sampling temperature")
        if not all(torch.isfinite(value).all() for value in self.parameters()):
            raise ValueError("Edge logits must be finite")
        terms = [sum(parameter.sum() for parameter in self.parameters()) * 0]

        def selected(logit):
            scaled = logit / temperature
            if mode == "fixed":
                return True
            if mode == "threshold":
                return bool(torch.sigmoid(scaled).item() >= threshold)
            keep = bool(torch.rand((), generator=generator).item() < torch.sigmoid(scaled).item())
            terms.append(F.logsigmoid(scaled if keep else -scaled))
            return keep

        rounds = []
        size = self.config.workers
        for round_index in range(self.config.rounds):
            spatial, temporal = [], []
            for source in range(size):
                for target in range(size):
                    if (self.spatial_mask[round_index, source, target]
                            and not _reaches(spatial, target, source)
                            and selected(self.spatial_logits[round_index, source, target])):
                        spatial.append([source, target])
                    if (round_index > 0 and self.temporal_mask[round_index - 1, source, target]
                            and selected(self.temporal_logits[round_index - 1, source, target])):
                        temporal.append([source, target])
            rounds.append({"spatial": spatial, "temporal": temporal})
        summary = [i for i in range(size) if self.summary_mask[i] and selected(self.summary_logits[i])]
        return SampledTopology({"rounds": rounds, "summary": summary}, torch.stack(terms).sum())

    def sparsity_loss(self):
        """Nuclear-norm relaxation of the rank penalty, applied to probabilities."""
        weights = self.weights()
        terms = [weights["spatial"].sum() * 0]
        for kind in ("spatial", "temporal"):
            if weights[kind].numel():
                terms.append(torch.linalg.svdvals(weights[kind]).sum(-1).mean())
        terms.append(weights["summary"].mean())
        return torch.stack(terms).sum()

    @torch.no_grad()
    def prune(self, rate: float):
        if not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("pruning rate must be in [0, 1]")
        report = {}
        for kind in ("spatial", "temporal", "summary"):
            masks = getattr(self, f"{kind}_mask")
            logits = getattr(self, f"{kind}_logits")
            if not torch.isfinite(logits).all():
                raise ValueError("Cannot prune nonfinite logits")
            matrices = [(masks, logits)] if kind == "summary" else list(zip(masks, logits))
            before = int(masks.sum())
            for mask, values in matrices:
                active = torch.where(mask.flatten())[0]
                count = math.floor(len(active) * rate)
                ranked = torch.argsort(values.flatten()[active], stable=True)
                mask.view(-1)[active[ranked[:count]]] = False
            report[kind] = {"before": before, "after": int(masks.sum())}
        return report

    def report(self):
        return {"config": asdict(self.config),
                "weights": {name: value.detach().tolist() for name, value in self.weights().items()}}


def save_policy(path: Path, policy: EdgePolicy, metadata: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "config": asdict(policy.config), "metadata": metadata,
               "state": {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_policy(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported topology checkpoint")
    policy = EdgePolicy(TopologyConfig(**payload["config"]))
    expected = policy.state_dict()
    state = payload["state"]
    if set(state) != set(expected):
        raise ValueError("Incomplete topology checkpoint")
    for name, value in state.items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError("Checkpoint tensor shape or dtype mismatch")
        if name.endswith("mask") and (value & ~expected[name]).any():
            raise ValueError("Checkpoint enables edges outside the candidate graph")
        if not torch.isfinite(value).all():
            raise ValueError("Checkpoint contains nonfinite values")
    policy.load_state_dict(state)
    return policy, payload["metadata"]
