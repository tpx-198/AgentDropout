"""离线验证提示词迁移、调用次数、异步路径及真实后端边界。"""

import ast
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langsmith import tracing_context

from ad_l.models import ModelAdapter, ModelSettings, create_model_adapter
from ad_l.prompts import build_math_solver_messages, load_math_solver_prompt
from ad_l.single_agent import build_single_agent_graph

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path("/Users/tpx-198/code/AgentDropout")


class SingleAgentTests(unittest.TestCase):
    def setUp(self):
        # 禁止测试建立网络连接，包括意外的模型或追踪请求。
        self.network = patch("socket.socket.connect", side_effect=AssertionError("offline test"))
        self.network.start()
        self.tracing = tracing_context(enabled=False)
        self.tracing.__enter__()

    def tearDown(self):
        self.tracing.__exit__(None, None, None)
        self.network.stop()

    def test_sync_async_and_invocation_isolation(self):
        adapter = create_model_adapter(mock_response="The answer is 5")
        with patch.object(adapter, "invoke", wraps=adapter.invoke) as invoke:
            graph = build_single_agent_graph(adapter)
            first = graph.invoke({"task": "3 + 2?"})
            second = graph.invoke({"task": "A different question"})
            self.assertEqual(invoke.call_count, 2)
        self.assertEqual(first["answer"], "The answer is 5")
        self.assertIsNone(first["usage"])
        self.assertEqual(len(second["messages"]), 2)
        self.assertNotIn("3 + 2?", second["messages"][1].content)
        with patch.object(adapter, "ainvoke", wraps=adapter.ainvoke) as invoke:
            result = asyncio.run(graph.ainvoke({"task": "3 + 2?"}))
            self.assertEqual(invoke.call_count, 1)
        self.assertEqual(result["answer"], first["answer"])

    def test_provider_usage_and_answer_are_preserved(self):
        response = AIMessage(content="The answer is 5", usage_metadata={
            "input_tokens": 20, "output_tokens": 5, "total_tokens": 25,
        })
        adapter = ModelAdapter(FakeMessagesListChatModel(responses=[response]), ModelSettings())
        result = build_single_agent_graph(adapter).invoke({"task": "3 + 2?"})
        self.assertEqual(result["usage"], response.usage_metadata)
        self.assertEqual(result["answer"], response.content)

    def test_invalid_task(self):
        for task in ["", "   ", None]:
            with self.assertRaises(ValueError):
                build_math_solver_messages(task)

    def test_live_requires_explicit_opt_in_and_key(self):
        settings = ModelSettings(backend="deepseek", model="test-model")
        with self.assertRaisesRegex(ValueError, "allow_real_api"):
            create_model_adapter(settings)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
                create_model_adapter(settings, allow_real_api=True)

    def test_live_configuration_without_request(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("DEEPSEEK_API_KEY=file-test-value\n")
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-test-value"}, clear=True):
                with patch("langchain_deepseek.ChatDeepSeek") as client:
                    create_model_adapter(ModelSettings(backend="deepseek", model="test-model"),
                                         allow_real_api=True, env_file=env_file)
                    self.assertEqual(client.call_args.kwargs["api_key"], "env-test-value")
                    self.assertIsNone(client.call_args.kwargs["temperature"])
                    self.assertIsNone(client.call_args.kwargs["max_tokens"])
                    client.return_value.invoke.assert_not_called()

    def test_mock_does_not_load_dotenv(self):
        with patch("dotenv.load_dotenv") as load:
            create_model_adapter(env_file=Path("missing.env"))
            load.assert_not_called()

    @unittest.skipUnless(REFERENCE.exists(), "原项目仅用于本地迁移对照")
    def test_exact_messages_against_original_method(self):
        prompt_path = REFERENCE / "AgentDropout/prompt/gsm8k_prompt_set.py"
        prompt_tree = ast.parse(prompt_path.read_text())
        values = {}
        for node in prompt_tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {"ROLE_DESCRIPTION", "FEW_SHOT_DATA"}:
                        values[target.id] = ast.literal_eval(node.value)
        agent_tree = ast.parse((REFERENCE / "AgentDropout/agents/math_solver.py").read_text())
        agent = next(n for n in agent_tree.body if isinstance(n, ast.ClassDef) and n.name == "MathSolver")
        method = next(n for n in agent.body if isinstance(n, ast.FunctionDef) and n.name == "_process_inputs")
        namespace = {"Dict": Dict, "List": List, "Any": Any}
        # 只执行原提示词构造函数，不导入原模型模块或原项目入口。
        exec(compile(ast.Module(body=[method], type_ignores=[]), "original_prompt", "exec"), namespace)
        role = "Math Solver"
        original_self = SimpleNamespace(
            role=role, constraint=values["ROLE_DESCRIPTION"][role],
            prompt_set=SimpleNamespace(get_answer_prompt=lambda question, role:
                                      f"{values['FEW_SHOT_DATA'][role]}\n\nQ:{question}"),
        )
        for task in ["3 + 2?", "有 {n} 个苹果\n第二行"]:
            expected = namespace["_process_inputs"](original_self, {"task": task}, {}, {})
            self.assertEqual(tuple(m.content for m in build_math_solver_messages(task)), expected)
        for relative, digest in load_math_solver_prompt()["source"]["files"].items():
            self.assertEqual(hashlib.sha256((REFERENCE / relative).read_bytes()).hexdigest(), digest)

    def test_cli_saves_offline_record(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.json"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/run_single_agent.py"),
                                     "--output", str(output)], capture_output=True, text=True,
                                    env={**os.environ, "LANGSMITH_TRACING": "false"}, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(output.read_text())
            self.assertEqual(record["real_api_calls"], 0)
            self.assertEqual(record["model_calls"], 1)
            self.assertEqual(record["data_split"], "manual")
            self.assertIsNone(record["usage"])
            self.assertEqual(record["usage_source"], "mock_not_measured")
            self.assertNotIn("api_key", json.dumps(record).lower())


if __name__ == "__main__":
    unittest.main()
