"""Numeric and multiple-choice rewards for REINFORCE edge training."""

import math
import re
from decimal import Decimal, DecimalException

import torch

from ad_l.roles import create_team
from ad_l.system import build_system
from ad_l.datasets import Example, load_examples


def answer_number(text: str):
    number = r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    number = number + r"(?:\s*/\s*" + number + r")?"
    matches = re.findall(r"(?:the answer is|####)\s*\$?(" + number + r")", text, flags=re.IGNORECASE)
    if not matches:
        matches = re.findall(r"\\boxed\{(" + number + r")\}", text)
    if not matches:
        return None
    try:
        parts = matches[-1].replace(",", "").split("/")
        value = Decimal(parts[0])
        if len(parts) == 2:
            value /= Decimal(parts[1])
    except DecimalException:
        return None
    return value if value.is_finite() else None


def score_example(answer: str, example: Example):
    if example.answer_type == "numeric":
        prediction = answer_number(answer)
        correct = prediction is not None and prediction == Decimal(example.target)
        return {"prediction": str(prediction) if prediction is not None else None, "correct": correct}
    if example.answer_type != "choice":
        raise ValueError("Unsupported answer type")
    matches = re.findall(r"(?:the answer is|answer\s*:|####)\s*(?:option\s*)?\(?([A-Z])\)?\s*[.!]?\s*$",
                         answer, flags=re.IGNORECASE | re.MULTILINE)
    if not matches:
        matches = re.findall(r"\\boxed\{([A-Z])\}", answer, flags=re.IGNORECASE)
    if not matches:
        matches = re.findall(r"^\s*([A-Z])\s*[.)]?\s*$", answer, flags=re.IGNORECASE | re.MULTILINE)
    prediction = matches[-1].upper() if matches else None
    if prediction is None or prediction not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(example.choices)]:
        prediction = None
    return {"prediction": prediction, "correct": prediction == example.target}


def train_step(policy, optimizer, examples, adapters, generator, *, sparsity=0.01, temperature=1.0,
               allow_real_api=False, roles=None):
    if not examples or not math.isfinite(sparsity) or sparsity < 0:
        raise ValueError("Provide a nonempty batch and a finite nonnegative sparsity coefficient")
    if any(example.split != "train" for example in examples):
        raise ValueError("Topology training accepts only training examples")
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
        state = graph.invoke({"task": example.task, "answer_type": example.answer_type})
        score = score_example(state["answer"], example) if state["status"] == "success" else {"prediction": None, "correct": False}
        reward = float(score["correct"])
        episodes.append({"sample_id": example.sample_id, "dataset": example.dataset, "split": example.split,
                         "answer_type": example.answer_type, "target": example.target, "reward": reward, **score,
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
