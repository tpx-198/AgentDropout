"""验证固定图的实际可见性、原项目逐调用对照及离线边界。"""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from langsmith import tracing_context

from ad_l.fixed_graph import (OfflineFixedGraphConfig, build_offline_fixed_graph,
                             load_fixed_graph_config, resolve_fixed_topology)
from ad_l.gsm8k_compat import gsm_get_predict
from ad_l.prompts import load_fixed_graph_prompts
from original_fixed_graph_reference import REFERENCE, original_namespace, run_original_fixed_graph

ROOT = Path(__file__).resolve().parents[1]


class FixedGraphTests(unittest.TestCase):
    def setUp(self):
        self.network = patch("socket.socket.connect", side_effect=AssertionError("offline test"))
        self.network.start()
        self.tracing = tracing_context(enabled=False)
        self.tracing.__enter__()
        self.config = load_fixed_graph_config(ROOT / "configs/fixed_graph_offline.json")

    def tearDown(self):
        self.tracing.__exit__(None, None, None)
        self.network.stop()

    def run_graph(self, config=None, task="3 + 2?"):
        return build_offline_fixed_graph(config or self.config).invoke({"task": task})

    def test_default_visibility_and_final_round_only(self):
        state = self.run_graph()
        a1, b1, a2, b2, final = state["trace"]
        self.assertEqual([(c["round"], c["node_id"]) for c in state["trace"]],
                         [(1, "A"), (1, "B"), (2, "A"), (2, "B"), (2, "final")])
        self.assertEqual(a1["spatial_info"], {})
        self.assertEqual(a1["temporal_info"], {})
        self.assertEqual(b1["temporal_info"], {})
        self.assertEqual(b1["spatial_info"]["A"]["output"], a1["output"])
        self.assertEqual(a2["spatial_info"], {})
        self.assertEqual(a2["temporal_info"], {})
        self.assertEqual(b2["spatial_info"]["A"]["output"], a2["output"])
        self.assertEqual(b2["temporal_info"]["A"]["output"], a1["output"])
        self.assertEqual(list(final["spatial_info"]), ["A", "B"])
        self.assertEqual(final["spatial_info"]["A"]["output"], a2["output"])
        self.assertEqual(final["spatial_info"]["B"]["output"], b2["output"])
        self.assertNotIn(b1["output"], final["messages"][1]["content"])
        self.assertEqual(final["temporal_info"], {})
        self.assertEqual(state["answer"], self.config.final_response)
        self.assertEqual(state["model_calls"], 5)
        self.assertEqual(state["real_api_calls"], 0)
        self.assertIsNone(state["usage"])
        self.assertEqual(len(state["round_outputs"]), 2)

    def test_removing_edges_removes_visibility(self):
        config = replace(self.config, spatial_masks=((0, 0), (0, 0)), temporal_masks=((0, 0), (0, 0)))
        state = self.run_graph(config)
        for call in state["trace"][:4]:
            self.assertEqual(call["spatial_info"], {})
            self.assertEqual(call["temporal_info"], {})
        self.assertEqual(len(state["trace"][-1]["spatial_info"]), 2)

    def test_previous_round_not_overwritten_by_current_outputs(self):
        config = replace(self.config, spatial_masks=((0, 0), (0, 0)))
        a1, b1, a2, b2, _ = self.run_graph(config)["trace"]
        self.assertEqual(a2["temporal_info"]["B"]["output"], b1["output"])
        self.assertEqual(b2["temporal_info"]["A"]["output"], a1["output"])
        self.assertNotEqual(b2["temporal_info"]["A"]["output"], a2["output"])

    def test_original_cycle_filter_on_fixed_edges(self):
        config = replace(self.config, spatial_masks=((1, 1), (1, 1)))
        topology = resolve_fixed_topology(config)
        self.assertEqual(topology["spatial_edges"], [["A", "B"]])
        self.assertEqual(topology["temporal_edges"], [["A", "B"]])

    def test_sync_async_repeat_and_concurrent_runs_are_isolated(self):
        graph = build_offline_fixed_graph(self.config)
        first = graph.invoke({"task": "FIRST_TASK"})
        second = graph.invoke({"task": "SECOND_TASK"})
        for call in second["trace"]:
            self.assertNotIn("FIRST_TASK", call["messages"][1]["content"])
        self.assertEqual(len(second["trace"]), 5)

        async def run():
            return await asyncio.gather(graph.ainvoke({"task": "FIRST_TASK"}),
                                        graph.ainvoke({"task": "SECOND_TASK"}))

        async_first, async_second = asyncio.run(run())
        self.assertEqual(first, async_first)
        self.assertEqual(second, async_second)

    def test_invalid_config_and_task_fail_before_model(self):
        with patch("ad_l.fixed_graph.create_model_adapter") as factory:
            for task in ["", "   ", None]:
                with self.assertRaises(ValueError):
                    self.run_graph(task=task)
            factory.assert_not_called()
        data = self.config.to_dict()
        for update in [{"backend": "deepseek"}, {"rounds": 3}, {"spatial_masks": [[0, 1]]},
                       {"spatial_masks": None}, {"spatial_masks": [0, 1]},
                       {"responses": {"A": ["a", "b"], "B": "ab"}},
                       {"responses": {"A": ["a", "b"]}}, {"final_response": ""},
                       {"agents": ["unexpected"]}]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                OfflineFixedGraphConfig.from_dict({**data, **update})
        with self.assertRaises(ValueError):
            OfflineFixedGraphConfig.from_dict([])

    def test_model_failure_does_not_produce_summary(self):
        with patch("ad_l.fixed_graph.create_model_adapter") as factory:
            factory.return_value.invoke.side_effect = RuntimeError("mock failure")
            with self.assertRaisesRegex(RuntimeError, "mock failure"):
                self.run_graph()
            self.assertEqual(factory.call_count, 1)

    @unittest.skipUnless(REFERENCE.exists(), "原项目仅用于本地迁移对照")
    def test_full_trace_matches_original_async_scheduler(self):
        variants = [self.config,
                    replace(self.config, spatial_masks=((0, 0), (1, 0))),
                    replace(self.config, spatial_masks=((1, 1), (1, 1))),
                    replace(self.config, spatial_masks=((0, 0), (0, 0))),
                    replace(self.config, temporal_masks=((0, 0), (0, 0)))]
        for config in variants:
            with self.subTest(topology=resolve_fixed_topology(config)):
                original = asyncio.run(run_original_fixed_graph(config, "中文题目 {x}\n3 + 2?"))
                actual = self.run_graph(config, "中文题目 {x}\n3 + 2?")
                self.assertEqual(actual["trace"], original.trace)
                self.assertEqual(actual["answer"], original.answer)
                self.assertEqual(original.log_prob.item(), 0)

    @unittest.skipUnless(REFERENCE.exists(), "原项目仅用于本地迁移对照")
    def test_original_hint_extractor_and_source_fingerprints(self):
        original = original_namespace()["gsm_get_predict"]
        for text in ["The answer is 5", "the answer is -12.", "The answer is 1.5",
                     r"Final: $\boxed{\frac{1}{2}}$", "No digits", "3 and 42"]:
            self.assertEqual(gsm_get_predict(text), original(text))
        for relative, digest in load_fixed_graph_prompts()["source"]["files"].items():
            self.assertEqual(hashlib.sha256((REFERENCE / relative).read_bytes()).hexdigest(), digest)

    def test_cli_sync_async_records(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in [[], ["--async"]]:
                output = Path(directory) / "result.json"
                result = subprocess.run([sys.executable, str(ROOT / "scripts/run_fixed_graph.py"),
                                         "--output", str(output), *mode], capture_output=True, text=True,
                                        env={**os.environ, "LANGSMITH_TRACING": "false"}, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                record = json.loads(output.read_text())
                self.assertEqual(record["model_calls"], 5)
                self.assertEqual(record["real_api_calls"], 0)
                self.assertEqual(record["usage_source"], "mock_not_measured")
                self.assertIsNone(record["usage"])
                self.assertEqual(record["effective_topology"]["temporal_edges"], [["A", "B"]])
                self.assertNotIn("api_key", json.dumps(record).lower())


if __name__ == "__main__":
    unittest.main()
