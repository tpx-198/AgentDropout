"""两个数学智能体、两轮固定通信图、FinalRefer；本阶段仅使用离线模型。"""

import json
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from ad_l.models import create_model_adapter
from ad_l.prompts import build_final_refer_messages, build_math_agent_messages

AGENTS = {"A": "Math Solver", "B": "Mathematical Analyst"}


@dataclass(frozen=True)
class OfflineFixedGraphConfig:
    spatial_masks: tuple
    temporal_masks: tuple
    responses: tuple
    final_response: str

    def __post_init__(self):
        for name in ("spatial_masks", "temporal_masks"):
            matrix = getattr(self, name)
            if (not isinstance(matrix, (list, tuple)) or len(matrix) != 2
                    or any(not isinstance(row, (list, tuple)) or len(row) != 2 for row in matrix)
                    or any(type(v) is not int or v not in (0, 1) for row in matrix for v in row)):
                raise ValueError(f"{name} 必须是 2×2 的 0/1 矩阵（行=来源，列=接收者）")
            object.__setattr__(self, name, tuple(tuple(row) for row in matrix))
        if (not isinstance(self.responses, (list, tuple)) or len(self.responses) != 2
                or any(not isinstance(row, (list, tuple)) or len(row) != 2 for row in self.responses)
                or any(not isinstance(v, str) or not v.strip() for row in self.responses for v in row)):
            raise ValueError("A 和 B 必须各提供两条非空模拟回答")
        object.__setattr__(self, "responses", tuple(tuple(row) for row in self.responses))
        if not isinstance(self.final_response, str) or not self.final_response.strip():
            raise ValueError("final_response 必须为非空字符串")

    @classmethod
    def from_dict(cls, data: dict):
        if not isinstance(data, dict):
            raise ValueError("固定图配置必须为 JSON 对象")
        allowed = {"backend", "rounds", "spatial_masks", "temporal_masks", "responses", "final_response"}
        if set(data) - allowed:
            raise ValueError(f"不支持的配置字段：{sorted(set(data) - allowed)}")
        if data.get("backend") != "mock" or type(data.get("rounds")) is not int or data["rounds"] != 2:
            raise ValueError("当前固定图仅支持 backend=mock、rounds=2")
        try:
            if set(data["responses"]) != set(AGENTS):
                raise ValueError("responses 只允许 A、B 两个智能体")
            return cls(data["spatial_masks"], data["temporal_masks"],
                       tuple(data["responses"][node_id] for node_id in AGENTS), data["final_response"])
        except (KeyError, TypeError) as error:
            raise ValueError("固定图配置缺少字段或字段类型不正确") from error

    def to_dict(self):
        return {"backend": "mock", "rounds": 2,
                "spatial_masks": [list(row) for row in self.spatial_masks],
                "temporal_masks": [list(row) for row in self.temporal_masks],
                "responses": {node_id: list(self.responses[i]) for i, node_id in enumerate(AGENTS)},
                "final_response": self.final_response}


def load_fixed_graph_config(path: Path) -> OfflineFixedGraphConfig:
    return OfflineFixedGraphConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))


def resolve_fixed_topology(config: OfflineFixedGraphConfig) -> dict:
    """复现原固定图的行优先建边和空间环路检查，含时间边的原有过滤行为。"""
    ids = list(AGENTS)
    spatial = []

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

    for i, source in enumerate(ids):
        for j, target in enumerate(ids):
            if config.spatial_masks[i][j] and not reaches(target, source):
                spatial.append((source, target))
    temporal = [(source, target) for i, source in enumerate(ids) for j, target in enumerate(ids)
                if config.temporal_masks[i][j] and not reaches(target, source)]
    indegree = {node: sum(dst == node for _, dst in spatial) for node in ids}
    pending = [node for node in ids if indegree[node] == 0]
    order = []
    while pending:
        node = pending.pop(0)
        order.append(node)
        for source, target in spatial:
            if source == node:
                indegree[target] -= 1
                if indegree[target] == 0:
                    pending.append(target)
    return {"spatial_edges": [list(edge) for edge in spatial],
            "temporal_edges": [list(edge) for edge in temporal],
            "execution_order": order}


class FixedGraphInput(TypedDict):
    task: str


class FixedGraphState(FixedGraphInput, total=False):
    current_outputs: dict
    previous_outputs: dict
    round_outputs: list[dict]
    trace: list[dict]
    answer: str
    usage: dict | None
    model_calls: int
    real_api_calls: int


def build_offline_fixed_graph(config: OfflineFixedGraphConfig):
    topology = resolve_fixed_topology(config)

    def initialize(state: FixedGraphInput):
        task = state.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task 必须为非空字符串")
        return {"current_outputs": {}, "previous_outputs": {}, "round_outputs": [],
                "trace": [], "answer": "", "usage": None, "model_calls": 0, "real_api_calls": 0}

    def visible(outputs, edges, target):
        return {source: {"role": AGENTS[source], "output": outputs[source]}
                for source in AGENTS if [source, target] in edges and source in outputs}

    def make_call_node(round_index, node_id):
        final = node_id == "final"

        def prepare(state):
            if final:
                spatial = {source: {"role": role, "output": state["previous_outputs"][source]}
                           for source, role in AGENTS.items()}
                temporal = {}
                messages = build_final_refer_messages(state["task"], spatial)
                text = config.final_response
            else:
                spatial = visible(state["current_outputs"], topology["spatial_edges"], node_id)
                temporal = visible(state["previous_outputs"], topology["temporal_edges"], node_id)
                messages = build_math_agent_messages(state["task"], AGENTS[node_id], spatial, temporal)
                text = config.responses[list(AGENTS).index(node_id)][round_index]
            # 每次调用单独创建模拟客户端，避免共享响应游标导致任务间或并发运行串线。
            adapter = create_model_adapter(mock_response=text)
            return messages, spatial, temporal, adapter

        def complete(state, messages, spatial, temporal, response: AIMessage):
            if not isinstance(response.content, str):
                raise TypeError("固定图仅支持文本回答")
            entry = {"stage": "final" if final else "agent", "round": round_index + 1,
                     "node_id": node_id, "role": "FinalRefer" if final else AGENTS[node_id],
                     "spatial_info": spatial, "temporal_info": temporal,
                     "messages": [{"role": m.type, "content": m.content} for m in messages],
                     "output": response.content, "usage": response.usage_metadata}
            trace = [*state["trace"], entry]
            update = {"trace": trace, "model_calls": len(trace)}
            if final:
                update["answer"] = response.content
                if all(call["usage"] is not None for call in trace):
                    update["usage"] = {key: sum(call["usage"][key] for call in trace)
                                       for key in ("input_tokens", "output_tokens", "total_tokens")}
            else:
                update["current_outputs"] = {**state["current_outputs"], node_id: response.content}
            return update

        def invoke(state: FixedGraphState):
            messages, spatial, temporal, adapter = prepare(state)
            return complete(state, messages, spatial, temporal, adapter.invoke(messages))

        async def ainvoke(state: FixedGraphState):
            messages, spatial, temporal, adapter = prepare(state)
            return complete(state, messages, spatial, temporal, await adapter.ainvoke(messages))

        return RunnableLambda(invoke, afunc=ainvoke)

    def commit_round(state: FixedGraphState):
        outputs = dict(state["current_outputs"])
        return {"previous_outputs": outputs, "current_outputs": {},
                "round_outputs": [*state["round_outputs"], outputs]}

    builder = StateGraph(FixedGraphState, input_schema=FixedGraphInput)
    builder.add_node("initialize", initialize)
    builder.add_edge(START, "initialize")
    previous = "initialize"
    for round_index in range(2):
        for node_id in topology["execution_order"]:
            name = f"round_{round_index + 1}_{node_id}"
            builder.add_node(name, make_call_node(round_index, node_id))
            builder.add_edge(previous, name)
            previous = name
        name = f"commit_round_{round_index + 1}"
        builder.add_node(name, commit_round)
        builder.add_edge(previous, name)
        previous = name
    builder.add_node("final", make_call_node(1, "final"))
    builder.add_edge(previous, "final")
    builder.add_edge("final", END)
    return builder.compile()
