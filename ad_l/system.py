"""LangGraph execution. Only plain topology data enters graph state."""

import copy
from dataclasses import asdict
from time import perf_counter

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from ad_l.models import ModelAdapter
from ad_l.roles import Agent, decision_messages, worker_messages
from ad_l.topology import topological_order


class TeamState(TypedDict, total=False):
    task: str
    current: dict
    previous: dict
    round_outputs: list
    trace: list
    answer: str
    status: str
    model_calls: int
    real_api_calls: int
    usage: dict | None


def validate_execution(execution: dict, workers: int):
    rounds = execution.get("rounds")
    summary = execution.get("summary")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("At least one discussion round is required")
    if (not isinstance(summary, list) or len(set(summary)) != len(summary)
            or any(type(node) is not int or not 0 <= node < workers for node in summary)):
        raise ValueError("Invalid final aggregation sources")
    orders = []
    for index, graph in enumerate(rounds):
        orders.append(topological_order(workers, graph["spatial"]))
        temporal = graph["temporal"]
        if not isinstance(temporal, list) or (index == 0 and temporal):
            raise ValueError("The first round cannot have previous-round edges")
        seen = set()
        for edge in temporal:
            if (not isinstance(edge, (list, tuple)) or len(edge) != 2
                    or any(type(node) is not int or not 0 <= node < workers for node in edge)
                    or tuple(edge) in seen):
                raise ValueError("Invalid or duplicate previous-round edge")
            seen.add(tuple(edge))
    return orders


def build_system(team: tuple[Agent, ...], execution: dict, adapters: dict[str, ModelAdapter], *, allow_real_api=False):
    if (len(team) < 2 or team[-1].role != "FinalRefer"
            or len({agent.node_id for agent in team}) != len(team)):
        raise ValueError("A team requires unique IDs and a final decision-maker")
    if set(adapters) != {agent.node_id for agent in team} or any(
            not isinstance(adapter, ModelAdapter) for adapter in adapters.values()):
        raise ValueError("Provide a ModelAdapter for every team member")
    if not allow_real_api and any(adapter.settings.backend != "mock" for adapter in adapters.values()):
        raise ValueError("Real model execution requires allow_real_api=True")
    workers = team[:-1]
    execution = copy.deepcopy(execution)
    adapters = dict(adapters)
    orders = validate_execution(execution, len(workers))

    def initialize(state):
        if not isinstance(state.get("task"), str) or not state["task"].strip():
            raise ValueError("task must be a nonempty string")
        return {"current": {}, "previous": {}, "round_outputs": [], "trace": [],
                "answer": "", "status": "running", "model_calls": 0, "real_api_calls": 0, "usage": None}

    def visible(outputs, sources):
        return {workers[source].node_id: {"role": workers[source].role,
                                         "output": outputs[workers[source].node_id]}
                for source in sources}

    def call_node(round_index, worker_index=None):
        final = worker_index is None
        agent = team[-1] if final else workers[worker_index]
        adapter = adapters[agent.node_id]

        def prepare(state):
            if final:
                spatial = visible(state["previous"], execution["summary"])
                temporal = {}
                messages = decision_messages(state["task"], spatial)
            else:
                graph = execution["rounds"][round_index]
                spatial = visible(state["current"], [source for source, target in graph["spatial"] if target == worker_index])
                temporal = visible(state["previous"], [source for source, target in graph["temporal"] if target == worker_index])
                messages = worker_messages(state["task"], agent.role, spatial, temporal)
            return messages, spatial, temporal

        def complete(state, messages, spatial, temporal, started, response=None, error=None):
            if error is None and (not isinstance(response, AIMessage) or not isinstance(response.content, str)):
                error = TypeError("Expected an AIMessage with textual content")
            entry = {"node_id": agent.node_id, "role": agent.role, "round": round_index + 1,
                     "stage": "final" if final else "discussion", "spatial": spatial, "temporal": temporal,
                     "messages": [{"role": message.type, "content": message.content} for message in messages],
                     "output": response.content if error is None else None,
                     "usage": response.usage_metadata if isinstance(response, AIMessage) else None,
                     "elapsed_seconds": perf_counter() - started, "settings": asdict(adapter.settings),
                     "real_api_attempted": adapter.settings.backend != "mock",
                     "error_type": type(error).__name__ if error else None}
            trace = [*state["trace"], entry]
            usage = None
            if all(call["usage"] is not None for call in trace):
                usage = {key: sum(call["usage"][key] for call in trace)
                         for key in ("input_tokens", "output_tokens", "total_tokens")}
            update = {"trace": trace, "model_calls": len(trace), "usage": usage,
                      "real_api_calls": sum(call["real_api_attempted"] for call in trace)}
            if error is not None:
                update["status"] = "failed"
            elif final:
                update.update(answer=response.content, status="success")
            else:
                update["current"] = {**state["current"], agent.node_id: response.content}
            return update

        def invoke(state):
            messages, spatial, temporal = prepare(state)
            started = perf_counter()
            try:
                response = adapter.invoke(messages)
            except Exception as error:
                return complete(state, messages, spatial, temporal, started, error=error)
            return complete(state, messages, spatial, temporal, started, response=response)

        async def ainvoke(state):
            messages, spatial, temporal = prepare(state)
            started = perf_counter()
            try:
                response = await adapter.ainvoke(messages)
            except Exception as error:
                return complete(state, messages, spatial, temporal, started, error=error)
            return complete(state, messages, spatial, temporal, started, response=response)

        return RunnableLambda(invoke, afunc=ainvoke)

    def commit(state):
        outputs = dict(state["current"])
        return {"previous": outputs, "current": {}, "round_outputs": [*state["round_outputs"], outputs]}

    builder = StateGraph(TeamState)
    builder.add_node("initialize", initialize)
    builder.add_edge(START, "initialize")
    previous = "initialize"
    for round_index, order in enumerate(orders):
        for worker_index in order:
            name = f"round_{round_index + 1}_{workers[worker_index].node_id}"
            builder.add_node(name, call_node(round_index, worker_index))
            builder.add_conditional_edges(previous, lambda state: state["status"] == "failed", {True: END, False: name})
            previous = name
        name = f"commit_{round_index + 1}"
        builder.add_node(name, commit)
        builder.add_conditional_edges(previous, lambda state: state["status"] == "failed", {True: END, False: name})
        previous = name
    builder.add_node("final", call_node(len(orders) - 1))
    builder.add_edge(previous, "final")
    builder.add_edge("final", END)
    return builder.compile().with_config({"recursion_limit": len(workers) * len(orders) + len(orders) + 5})
