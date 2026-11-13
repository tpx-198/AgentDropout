"""GSM8K rewards and REINFORCE training of edge probabilities."""

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch

from ad_l.roles import create_team
from ad_l.system import build_system


@dataclass(frozen=True)
class Example:
    sample_id: str
    task: str
    target: str


def answer_number(text: str):
    number = r"[-+]?\d[\d,]*(?:\.\d+)?"
    matches = re.findall(r"(?:the answer is|####)\s*\$?(" + number + r")", text, flags=re.IGNORECASE)
    if not matches:
        matches = re.findall(r"\\boxed\{(" + number + r")\}", text)
    if not matches:
        return None
    try:
        value = Decimal(matches[-1].replace(",", ""))
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def load_examples(path: Path, limit: int, seed: int):
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be positive")
    raw = Path(path).read_bytes()
    records = []
    for index, line in enumerate(raw.decode("utf-8").splitlines()):
        record = json.loads(line)
        task, answer = record["question"], record["answer"]
        if not isinstance(task, str) or not task.strip() or not isinstance(answer, str) or "####" not in answer:
            raise ValueError(f"Invalid GSM8K row {index + 1}")
        target = answer.rsplit("####", 1)[1].strip().replace(",", "")
        if not Decimal(target).is_finite():
            raise ValueError("Expected a finite numeric target")
        records.append(Example(f"gsm8k:train:{index}", task, target))
    if limit > len(records):
        raise ValueError("limit exceeds the training dataset size")
    indices = sorted(random.Random(seed).sample(range(len(records)), limit))
    return [records[index] for index in indices], {"path": str(Path(path).resolve()), "split": "train",
        "sha256": hashlib.sha256(raw).hexdigest(), "records": len(records)}


def train_step(policy, optimizer, examples, adapters, generator, *, sparsity=0.01, temperature=1.0,
               allow_real_api=False, roles=None):
    if not examples or not math.isfinite(sparsity) or sparsity < 0:
        raise ValueError("Provide a nonempty batch and a finite nonnegative sparsity coefficient")
    if {id(value) for value in policy.parameters()} != {
            id(value) for group in optimizer.param_groups for value in group["params"]}:
        raise ValueError("Optimizer parameters must match the policy")
    team = create_team(policy.config.agents, roles)
    rng_before = generator.get_state()
    losses, episodes = [], []
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    for example in examples:
        sampled = policy.sample(generator, temperature=temperature)
        graph = build_system(team, sampled.execution, adapters, allow_real_api=allow_real_api)
        state = graph.invoke({"task": example.task})
        reward = float(answer_number(state["answer"]) == Decimal(example.target)) if state["status"] == "success" else 0.0
        episodes.append({"sample_id": example.sample_id, "target": example.target, "reward": reward,
                         "execution": sampled.execution, "result": state})
        if state["status"] != "success":
            generator.set_state(rng_before)
            return {"status": "failed", "episodes": episodes, "updated": False}
        losses.append(-sampled.log_prob * reward)
    policy_loss = torch.stack(losses).mean()
    regularization = policy.sparsity_loss()
    loss = policy_loss + sparsity * regularization
    loss.backward()
    if not torch.isfinite(loss) or not all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in policy.parameters()):
        optimizer.zero_grad(set_to_none=True)
        generator.set_state(rng_before)
        raise RuntimeError("Nonfinite training loss or gradient")
    before = [value.detach().clone() for value in policy.parameters()]
    optimizer.step()
    return {"status": "success", "updated": any(not torch.equal(old, new) for old, new in zip(before, policy.parameters())),
            "loss": loss.detach().item(), "policy_loss": policy_loss.detach().item(),
            "sparsity_loss": regularization.detach().item(), "episodes": episodes}
