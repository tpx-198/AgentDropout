"""Original AgentDropout GSM8K roles and message construction."""

import json
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

PROMPTS = json.loads((Path(__file__).parent / "resources/prompts.json").read_text(encoding="utf-8"))
ROLE_CYCLE = tuple(PROMPTS["roles"])


@dataclass(frozen=True)
class Agent:
    node_id: str
    role: str


def create_team(total_agents: int = 5, roles: list[str] | None = None) -> tuple[Agent, ...]:
    """The final member is the decision-maker and runs after all discussion rounds."""
    if type(total_agents) is not int or total_agents < 2:
        raise ValueError("At least two agents are required, including the decision-maker")
    worker_count = total_agents - 1
    selected = roles if roles is not None else [ROLE_CYCLE[i % len(ROLE_CYCLE)] for i in range(worker_count)]
    if len(selected) != worker_count or any(role not in PROMPTS["roles"] for role in selected):
        raise ValueError(f"Provide {worker_count} roles selected from {ROLE_CYCLE}")
    workers = tuple(Agent(f"agent_{i + 1}", role) for i, role in enumerate(selected))
    return (*workers, Agent("final", "FinalRefer"))


def worker_messages(task: str, role: str, spatial: dict, temporal: dict):
    prompt = PROMPTS["roles"][role]
    user = f"{prompt['few_shot']}\n\nQ:{task}"
    for label, visible in (("Current round", spatial), ("Previous round", temporal)):
        if visible:
            text = "\n\n".join(
                f"Agent {node_id} as {info['role']}\n{info['output']}"
                for node_id, info in visible.items()
            )
            user += f"\n\n{label} responses for reference\n{text}"
    return [SystemMessage(content=prompt["system"]), HumanMessage(content=user)]


def decision_messages(task: str, visible: dict):
    prompt = PROMPTS["decision"]
    text = "\n\n".join(f"{node_id}: {info['output']}" for node_id, info in visible.items())
    user = f"{prompt['few_shot']}\nThe task is\n{task}\nFinal-round agent outputs\n{text}"
    return [SystemMessage(content=prompt["system"]), HumanMessage(content=user)]
