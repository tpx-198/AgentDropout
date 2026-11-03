"""第一阶段仅迁移 GSM8K Math Solver、单轮、无前驱消息的提示词。"""

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage


def load_math_solver_prompt() -> dict:
    path = Path(__file__).with_name("resources") / "gsm8k_math_solver.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_math_solver_messages(task: str) -> list:
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task 必须为非空字符串")
    prompt = load_math_solver_prompt()
    # 保留原 MathSolver._process_inputs 在空 spatial/temporal 输入时的确切行为。
    user_prompt = f"{prompt['few_shot']}\n\nQ:{task}"
    user_prompt += "(Hint: The answer is near to)."
    return [SystemMessage(content=prompt["system"]), HumanMessage(content=user_prompt)]
