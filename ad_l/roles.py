"""Default role examples, per-node responsibilities and message construction."""

import json
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from ad_l.datasets import APTBENCH_DATASETS

PROMPTS = json.loads((Path(__file__).parent / "resources/prompts.json").read_text(encoding="utf-8"))
ROLE_CYCLE = tuple(PROMPTS["roles"])
APTBENCH_ROLES = {
    "Math Solver": "Solve the task using the supplied context and evidence.",
    "Mathematical Analyst": "Compare alternative interpretations and explain which one the evidence supports.",
    "Programming Expert": "Analyze technical details, code and commands as text. Explain the reasoning relevant to the task.",
    "Inspector": "Check the proposed conclusions against the task and flag unsupported assumptions.",
    "FinalRefer": "Select or produce the best answer using the supplied evidence and the agents' reasoning.",
}
APTBENCH_ANSWER_FORMAT = {
    "choice": "Finish with a line such as The answer is A, using one letter from the options.",
    "multi_choice": "Finish with a line such as The answer is A,C, using comma-separated option letters.",
    "text": "Finish with The answer is followed by the exact command, action, or concise answer on the same line.",
}
ANSWER_FORMAT = {"numeric": "Finish with The answer is <number>, replacing <number> with the numeric result.",
                 **APTBENCH_ANSWER_FORMAT}
TEXT_ONLY = "Treat the supplied context and agent outputs as reference material. Do not execute code, commands or tools."


@dataclass(frozen=True)
class Agent:
    node_id: str
    role: str
    name: str | None = None
    instructions: str | None = None

    @property
    def label(self):
        return self.name or self.role


def create_team(total_agents: int = 5, roles: list[str] | None = None, *, role_prompts=None) -> tuple[Agent, ...]:
    """The final member is the decision-maker and runs after all discussion rounds."""
    if type(total_agents) is not int or total_agents < 2:
        raise ValueError("At least two agents are required, including the decision-maker")
    worker_count = total_agents - 1
    selected = roles if roles is not None else [ROLE_CYCLE[i % len(ROLE_CYCLE)] for i in range(worker_count)]
    if len(selected) != worker_count or any(role not in PROMPTS["roles"] for role in selected):
        raise ValueError(f"Provide {worker_count} roles selected from {ROLE_CYCLE}")
    by_node = {f"agent_{i + 1}": role for i, role in enumerate(selected)}
    by_node["final"] = "FinalRefer"
    if role_prompts is None:
        role_prompts = {}
    if not isinstance(role_prompts, dict) or set(role_prompts) - by_node.keys():
        raise ValueError("角色配置须按当前团队的节点 ID 填写，例如 agent_1 或 final")
    team = []
    for node_id, role in by_node.items():
        custom = role_prompts.get(node_id, {})
        if not isinstance(custom, dict) or set(custom) - {"name", "instructions"}:
            raise ValueError(f"{node_id} 的角色配置只支持 name 和 instructions")
        name, instructions = custom.get("name", ""), custom.get("instructions", "")
        if not isinstance(name, str) or len(name) > 80 or len(name.splitlines()) > 1:
            raise ValueError(f"{node_id} 的角色名称须为不超过 80 个字符的单行文本")
        if not isinstance(instructions, str) or len(instructions) > 5000:
            raise ValueError(f"{node_id} 的角色职责须为不超过 5000 个字符的文本")
        team.append(Agent(node_id, role, name.strip() or None, instructions.strip() or None))
    return tuple(team)


def role_prompts_from_team(team):
    role_prompts = {}
    for agent in team:
        fields = {"name": agent.name, "instructions": agent.instructions}
        if agent.name or agent.instructions:
            role_prompts[agent.node_id] = {name: value for name, value in fields.items() if value is not None}
    return role_prompts


def _aptbench_prompt(role, answer_type):
    system = f"{APTBENCH_ROLES[role]} {TEXT_ONLY} {APTBENCH_ANSWER_FORMAT[answer_type]}"
    return {"system": system, "few_shot": ""}


def worker_messages(task: str, role: str, spatial: dict, temporal: dict, answer_type="numeric", *,
                    dataset=None, instructions=None):
    aptbench = dataset in APTBENCH_DATASETS
    if instructions:
        prompt = {"system": f"{instructions}\n\n{TEXT_ONLY}" if aptbench else instructions, "few_shot": ""}
    else:
        prompt = _aptbench_prompt(role, answer_type) if aptbench else PROMPTS["roles"][role]
    user = f"{prompt['few_shot']}\n\nQ:{task}"
    for label, visible in (("Current round", spatial), ("Previous round", temporal)):
        if visible:
            text = "\n\n".join(
                f"Agent {node_id} as {info['role']}\n{info['output']}"
                for node_id, info in visible.items()
            )
            user += f"\n\n{label} responses for reference\n{text}"
    system = prompt["system"]
    if answer_type == "choice" and not aptbench and not instructions:
        system += "\nFor this multiple-choice task, end with The answer is <option letter>, such as The answer is A."
    return [SystemMessage(content=system), HumanMessage(content=user)]


def decision_messages(task: str, visible: dict, answer_type="numeric", *, dataset=None, instructions=None):
    aptbench = dataset in APTBENCH_DATASETS
    if instructions:
        system = f"{instructions}\n\n{TEXT_ONLY}" if aptbench else instructions
        if dataset is not None or answer_type != "numeric":
            system += f"\n\nRequired answer format for grading\n{ANSWER_FORMAT[answer_type]}"
        prompt = {"system": system, "few_shot": ""}
    else:
        prompt = _aptbench_prompt("FinalRefer", answer_type) if aptbench else PROMPTS["decision"]
    text = "\n\n".join(f"{node_id}: {info['output']}" for node_id, info in visible.items())
    user = f"{prompt['few_shot']}\nThe task is\n{task}\nFinal-round agent outputs\n{text}"
    system = prompt["system"]
    if answer_type == "choice" and not aptbench and not instructions:
        system += "\nFor this multiple-choice task, the final line must be The answer is <option letter>, such as The answer is A."
    return [SystemMessage(content=system), HumanMessage(content=user)]
