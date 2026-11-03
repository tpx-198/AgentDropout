"""GSM8K 单智能体与固定图提示词；保留原项目的字符串拼接。"""

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from ad_l.gsm8k_compat import gsm_get_predict

PROMPT_DIR = Path(__file__).with_name("resources") / "prompts"


def load_math_solver_prompt() -> dict:
    path = PROMPT_DIR / "math_solver_prompts.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_math_solver_messages(task: str) -> list:
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task 必须为非空字符串")
    prompt = load_math_solver_prompt()
    # 保留原 MathSolver._process_inputs 在空 spatial/temporal 输入时的确切行为。
    user_prompt = f"{prompt['few_shot']}\n\nQ:{task}"
    user_prompt += "(Hint: The answer is near to)."
    return [SystemMessage(content=prompt["system"]), HumanMessage(content=user_prompt)]


def load_fixed_graph_prompts() -> dict:
    path = PROMPT_DIR / "math_collaboration_prompts.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_math_agent_messages(
    task: str, role: str, spatial_info: dict, temporal_info: dict,
) -> list:
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task 必须为非空字符串")
    prompt = load_fixed_graph_prompts()["roles"][role]
    user = f"{prompt['few_shot']}\n\nQ:{task}"
    if role == "Math Solver":
        user += "(Hint: The answer is near to"
        for info in spatial_info.values():
            user += " " + gsm_get_predict(info["output"])
        for info in temporal_info.values():
            user += " " + gsm_get_predict(info["output"])
        user += ")."
    else:
        spatial = "".join(
            f"Agent {node_id} as a {info['role']} his answer to this question is:\n\n{info['output']}\n\n"
            for node_id, info in spatial_info.items()
        )
        temporal = "".join(
            f"Agent {node_id} as a {info['role']} his answer to this question was:\n\n{info['output']}\n\n"
            for node_id, info in temporal_info.items()
        )
        if spatial:
            user += f"At the same time, there are the following responses to the same question for your reference:\n\n{spatial} \n\n"
        if temporal:
            user += f"In the last round of dialogue, there were the following responses to the same question for your reference: \n\n{temporal}"
    return [SystemMessage(content=prompt["system"]), HumanMessage(content=user)]


def build_final_refer_messages(task: str, spatial_info: dict) -> list:
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task 必须为非空字符串")
    prompt = load_fixed_graph_prompts()["decision"]
    system = f"{prompt['role']}.\n {prompt['constraint']}"
    spatial = "".join(node_id + ": " + info["output"] + "\n\n"
                      for node_id, info in spatial_info.items())
    user = f"{prompt['few_shot']} The task is:\n\n {task}.\n At the same time, the output of other agents is as follows:\n\n{spatial}"
    return [SystemMessage(content=system), HumanMessage(content=user)]
