"""两个数学智能体、两轮固定通信图、FinalRefer，支持注入模型适配器。"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Mapping

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from ad_l.models import ModelAdapter, ModelSettings, create_model_adapter
from ad_l.prompts import build_final_refer_messages, build_math_agent_messages
from ad_l.communication import execution_topology

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
    status: str
    known_usage: dict
    usage_complete: bool


def build_fixed_graph(
    config: OfflineFixedGraphConfig,
    adapters: Mapping[str, ModelAdapter] | None = None,
    *,
    allow_real_api: bool = False,
    round_topologies: list[dict] | None = None,
):
    """adapters 的键为 A、B、final；省略时按原预设响应离线运行。"""
    adapters = dict(adapters) if adapters is not None else None
    if adapters is not None:
        if set(adapters) != {*AGENTS, "final"}:
            raise ValueError("注入适配器必须同时提供 A、B、final")
        for adapter in adapters.values():
            if not isinstance(adapter, ModelAdapter):
                raise TypeError("必须注入 ModelAdapter 对象")
            if adapter.settings.backend != "mock" and not allow_real_api:
                raise ValueError("真实模型请求需显式设置 allow_real_api=True")
            if adapter.settings.max_retries != 0:
                raise ValueError("固定图当前要求 max_retries=0，以明确记录每次调用尝试")
    if round_topologies is None:
        topology = resolve_fixed_topology(config)
        topologies = [topology, topology]
    else:
        if len(round_topologies) != 2:
            raise ValueError("必须提供两轮普通连接数据")
        topologies = [execution_topology(item["spatial_edges"], item["temporal_edges"])
                      for item in round_topologies]
        if topologies[0]["temporal_edges"]:
            raise ValueError("第一轮不能有时间边")

    def initialize(state: FixedGraphInput):
        task = state.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task 必须为非空字符串")
        return {"current_outputs": {}, "previous_outputs": {}, "round_outputs": [],
                "trace": [], "answer": "", "usage": None, "model_calls": 0, "real_api_calls": 0,
                "status": "running", "known_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "usage_complete": False}

    def visible(outputs, edges, target):
        return {source: {"role": AGENTS[source], "output": outputs[source]}
                for source in AGENTS if [source, target] in edges and source in outputs}

    def make_call_node(round_index, node_id):
        final = node_id == "final"

        def prepare(state):
            topology = topologies[round_index]
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
            adapter = adapters[node_id] if adapters is not None else create_model_adapter(mock_response=text)
            return messages, spatial, temporal, adapter

        def complete(state, messages, spatial, temporal, adapter, started_at, started, response=None, error=None):
            usage = response.usage_metadata if isinstance(response, AIMessage) else None
            if error is None and (not isinstance(response, AIMessage) or not isinstance(response.content, str)):
                error = TypeError("固定图仅支持 AIMessage 文本回答")
            entry = {"stage": "final" if final else "agent", "round": round_index + 1,
                     "node_id": node_id, "role": "FinalRefer" if final else AGENTS[node_id],
                     "spatial_info": spatial, "temporal_info": temporal,
                     "messages": [{"role": m.type, "content": m.content} for m in messages],
                     "output": response.content if error is None else None, "usage": usage,
                     "settings": asdict(adapter.settings), "started_at_utc": started_at,
                     "elapsed_seconds": perf_counter() - started,
                     "status": "failed" if error else "success",
                     # 异常原文可能包含请求头或凭据，不写入结果或终端。
                     "error": {"type": type(error).__name__, "message": "模型调用或回答处理失败",
                               "status_code": getattr(error, "status_code", None)
                               if type(getattr(error, "status_code", None)) is int else None} if error else None,
                     "usage_source": "mock_not_measured" if adapter.settings.backend == "mock" else (
                         "provider" if usage is not None else "provider_unavailable"),
                     "real_api_attempted": adapter.settings.backend != "mock"}
            trace = [*state["trace"], entry]
            known_usage = {key: sum(call["usage"][key] for call in trace if call["usage"] is not None)
                           for key in ("input_tokens", "output_tokens", "total_tokens")}
            complete_usage = all(call["usage"] is not None for call in trace)
            update = {"trace": trace, "model_calls": len(trace), "known_usage": known_usage,
                      "usage_complete": complete_usage,
                      "usage": known_usage if complete_usage else None,
                      "real_api_calls": sum(call["real_api_attempted"] for call in trace)}
            if error is not None:
                update["status"] = "failed"
            elif final:
                update["answer"] = response.content
                update["status"] = "success"
            else:
                update["current_outputs"] = {**state["current_outputs"], node_id: response.content}
            return update

        def invoke(state: FixedGraphState):
            messages, spatial, temporal, adapter = prepare(state)
            started_at, started = datetime.now(timezone.utc).isoformat(), perf_counter()
            try:
                response = adapter.invoke(messages)
            except Exception as error:
                return complete(state, messages, spatial, temporal, adapter, started_at, started, error=error)
            return complete(state, messages, spatial, temporal, adapter, started_at, started, response=response)

        async def ainvoke(state: FixedGraphState):
            messages, spatial, temporal, adapter = prepare(state)
            started_at, started = datetime.now(timezone.utc).isoformat(), perf_counter()
            try:
                response = await adapter.ainvoke(messages)
            except Exception as error:
                return complete(state, messages, spatial, temporal, adapter, started_at, started, error=error)
            return complete(state, messages, spatial, temporal, adapter, started_at, started, response=response)

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
        for node_id in topologies[round_index]["execution_order"]:
            name = f"round_{round_index + 1}_{node_id}"
            builder.add_node(name, make_call_node(round_index, node_id))
            if previous.startswith("round_"):
                builder.add_conditional_edges(previous, lambda state: state["status"] == "failed",
                                              {True: END, False: name})
            else:
                builder.add_edge(previous, name)
            previous = name
        name = f"commit_round_{round_index + 1}"
        builder.add_node(name, commit_round)
        builder.add_conditional_edges(previous, lambda state: state["status"] == "failed",
                                      {True: END, False: name})
        previous = name
    builder.add_node("final", make_call_node(1, "final"))
    builder.add_edge(previous, "final")
    builder.add_edge("final", END)
    return builder.compile()


def build_offline_fixed_graph(config: OfflineFixedGraphConfig):
    """保留原离线入口，共用可注入模型的执行流程。"""
    return build_fixed_graph(config)
