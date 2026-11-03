"""只抽取原项目的纯定义，在 ad-l 中接入桩响应运行原调度器；不导入原模型模块。"""

import ast
import asyncio
import copy
import re
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace

import torch

REFERENCE = Path("/Users/tpx-198/code/AgentDropout")


def parse(relative):
    # 原文件中存在未转义的反斜杠；这里只忽略解析旧源码时已知的弃用提示。
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid escape sequence", category=DeprecationWarning)
        return ast.parse((REFERENCE / relative).read_text(encoding="utf-8"))


def execute_nodes(nodes, namespace):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, "original_offline_reference", "exec"), namespace)


def original_namespace():
    namespace = {"ABC": ABC, "abstractmethod": abstractmethod, "asyncio": asyncio,
                 "torch": torch, "re": re}
    prompt_tree = parse("AgentDropout/prompt/gsm8k_prompt_set.py")
    for node in prompt_tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"ROLE_DESCRIPTION", "FEW_SHOT_DATA"}:
                    namespace[target.id] = ast.literal_eval(node.value)
    prompt_class = copy.deepcopy(next(n for n in prompt_tree.body if isinstance(n, ast.ClassDef)))
    prompt_class.bases = []
    prompt_class.decorator_list = []
    execute_nodes([prompt_class], namespace)
    helper_names = {"gsm_get_predict", "_strip_string", "_remove_right_units", "_fix_sqrt",
                    "_fix_fracs", "_fix_a_slash_b"}
    execute_nodes([n for n in parse("datasets/gsm8k_dataset.py").body
                   if isinstance(n, ast.FunctionDef) and n.name in helper_names], namespace)
    for relative, name in [("AgentDropout/graph/node.py", "Node"),
                           ("AgentDropout/graph/graph.py", "Graph")]:
        execute_nodes([next(n for n in parse(relative).body if isinstance(n, ast.ClassDef)
                            and n.name == name)], namespace)
    for relative, cls_name, alias in [
        ("AgentDropout/agents/math_solver.py", "MathSolver", "math_prompt"),
        ("AgentDropout/agents/final_decision.py", "FinalRefer", "final_prompt"),
    ]:
        cls = next(n for n in parse(relative).body if isinstance(n, ast.ClassDef) and n.name == cls_name)
        method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                                    and n.name == "_process_inputs"))
        method.name = alias
        execute_nodes([method], namespace)
    return namespace


async def run_original_fixed_graph(config, task):
    namespace = original_namespace()
    trace = []

    class StubNode(namespace["Node"]):
        def __init__(self, node_id, role, responses, final=False):
            super().__init__(node_id)
            self.role = role
            self.prompt_set = namespace["GSM8KPromptSet"]
            self.constraint = self.prompt_set.get_constraint(role) if not final else ""
            self.responses = iter(responses)
            self.final = final
            self.calls = 0

        def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
            method = namespace["final_prompt" if self.final else "math_prompt"]
            return method(self, raw_inputs, spatial_info, temporal_info)

        def _execute(self, input, spatial_info, temporal_info, **kwargs):
            self.calls += 1
            system, user = self._process_inputs(input, spatial_info, temporal_info)
            response = next(self.responses)
            trace.append({"stage": "final" if self.final else "agent",
                          "round": 2 if self.final else self.calls, "node_id": self.id,
                          "role": "FinalRefer" if self.final else self.role,
                          "spatial_info": spatial_info, "temporal_info": temporal_info,
                          "messages": [{"role": "system", "content": system},
                                       {"role": "human", "content": user}],
                          "output": response, "usage": None})
            return response

        async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
            return self._execute(input, spatial_info, temporal_info)

    graph = namespace["Graph"].__new__(namespace["Graph"])
    graph.nodes = {"A": StubNode("A", "Math Solver", config.responses[0]),
                   "B": StubNode("B", "Mathematical Analyst", config.responses[1])}
    graph.decision_node = StubNode("final", "FinalRefer", [config.final_response], final=True)
    graph.potential_spatial_edges = [(src, dst) for src in graph.nodes for dst in graph.nodes]
    graph.potential_temporal_edges = list(graph.potential_spatial_edges)
    graph.spatial_masks = torch.tensor(config.spatial_masks).view(-1)
    graph.temporal_masks = torch.tensor(config.temporal_masks).view(-1)
    graph.spatial_logits = torch.ones(4) * 10
    graph.temporal_logits = torch.ones(4) * 10
    graph.optimized_spatial = graph.optimized_temporal = graph.diff = False
    graph.skip_nodes = []
    answers, log_prob = await graph.arun({"task": task}, num_rounds=2, max_tries=1)
    return SimpleNamespace(trace=trace, answer=answers[0], log_prob=log_prob)
