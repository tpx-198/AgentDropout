"""单轮数学智能体：构建提示词 → 模型 → 原样返回回答。"""

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from ad_l.models import ModelAdapter
from ad_l.prompts import build_math_solver_messages


class AgentState(TypedDict, total=False):
    task: str
    messages: list[BaseMessage]
    response: AIMessage
    answer: str
    usage: dict | None


def build_single_agent_graph(adapter: ModelAdapter):
    def prepare(state: AgentState):
        return {"messages": build_math_solver_messages(state["task"])}

    def generate(state: AgentState):
        return {"response": adapter.invoke(state["messages"])}

    async def agenerate(state: AgentState):
        return {"response": await adapter.ainvoke(state["messages"])}

    def finish(state: AgentState):
        response = state["response"]
        if not isinstance(response.content, str):
            raise TypeError("当前数学智能体仅支持文本回答")
        return {"answer": response.content, "usage": response.usage_metadata}

    from langchain_core.runnables import RunnableLambda

    builder = StateGraph(AgentState)
    builder.add_node("prepare", prepare)
    builder.add_node("generate", RunnableLambda(generate, afunc=agenerate))
    builder.add_node("finish", finish)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "generate")
    builder.add_edge("generate", "finish")
    builder.add_edge("finish", END)
    return builder.compile()
