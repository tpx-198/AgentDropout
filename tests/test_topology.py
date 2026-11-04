"""验证采样迁移、梯度回传、按轮执行和一次离线更新。"""

import json
import io
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

import torch
from langsmith import tracing_context

from ad_l.fixed_graph import build_fixed_graph, load_fixed_graph_config
from ad_l.topology import TopologyPolicy, execution_topology
from ad_l.topology_training import offline_training_step
from original_fixed_graph_reference import REFERENCE, original_namespace, parse, execute_nodes

ROOT = Path(__file__).resolve().parents[1]
FULL = ((1, 1), (1, 1))
OFF = ((0, 0), (0, 0))


class TopologyTests(unittest.TestCase):
    def setUp(self):
        self.network = patch("socket.socket.connect", side_effect=AssertionError("offline test"))
        self.network.start()
        self.tracing = tracing_context(enabled=False)
        self.tracing.__enter__()
        self.config = load_fixed_graph_config(ROOT / "configs/fixed_graph_offline.json")

    def tearDown(self):
        self.tracing.__exit__(None, None, None)
        self.network.stop()

    def test_samples_reproducible_acyclic_and_temporal_can_have_self_edges(self):
        policy = TopologyPolicy(FULL, FULL)
        first = policy.sample_rounds(generator=torch.Generator().manual_seed(0))
        second = policy.sample_rounds(generator=torch.Generator().manual_seed(0))
        self.assertEqual([r.for_report() for r in first], [r.for_report() for r in second])
        self.assertEqual(first[0].temporal_edges, ())
        for round_sample in first:
            round_sample.for_execution()
            self.assertTrue(round_sample.log_prob.requires_grad)
            self.assertNotIn(("A", "A"), round_sample.spatial_edges)
            self.assertNotIn(("B", "B"), round_sample.spatial_edges)
        self.assertTrue(any(src == dst for src, dst in first[1].temporal_edges))

    def test_policy_gradient_matches_bernoulli_formula_and_masked_entries_are_zero(self):
        policy = TopologyPolicy(((0, 1), (0, 0)), FULL)
        sampled = policy.sample_rounds(temperature=2, generator=torch.Generator().manual_seed(5))
        loss = -sum(item.log_prob for item in sampled)
        loss.backward()
        expected = {"spatial": torch.zeros(2, 2), "temporal": torch.zeros(2, 2)}
        for item in sampled:
            for decision in item.decisions:
                i, j = (0 if decision[key] == "A" else 1 for key in ("source", "target"))
                expected[decision["kind"]][i, j] += (decision["probability"] - float(decision["selected"])) / 2
        torch.testing.assert_close(policy.spatial_logits.grad, expected["spatial"])
        torch.testing.assert_close(policy.temporal_logits.grad, expected["temporal"])
        empty = TopologyPolicy(OFF, OFF)
        empty.sample_rounds()[0].log_prob.backward()
        self.assertTrue(torch.equal(empty.spatial_logits.grad, torch.zeros(2, 2)))

    def test_sampled_rounds_control_actual_visibility_and_do_not_filter_temporal_again(self):
        topologies = [execution_topology([["A", "B"]], []),
                      execution_topology([["B", "A"]], [["A", "A"], ["B", "A"], ["A", "B"]])]
        state = build_fixed_graph(self.config, round_topologies=topologies).invoke({"task": "3 + 2?"})
        a1, b1, b2, a2, final = state["trace"]
        self.assertEqual(b2["node_id"], "B")
        self.assertEqual(a2["spatial_info"]["B"]["output"], b2["output"])
        self.assertEqual(a2["temporal_info"]["A"]["output"], a1["output"])
        self.assertEqual(a2["temporal_info"]["B"]["output"], b1["output"])
        self.assertEqual(final["spatial_info"]["A"]["output"], a2["output"])
        json.dumps(state)  # LangGraph state 内没有训练张量

    def test_one_step_keeps_policy_gradient_and_changes_parameters(self):
        policy = TopologyPolicy(self.config.spatial_masks, self.config.temporal_masks)
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.1)
        report = offline_training_step(policy, optimizer, self.config,
            [{"sample_id": "synthetic:0", "task": "3 + 2?", "target": "5"}],
            generator=torch.Generator().manual_seed(0))
        self.assertEqual(report["rewards"], [1.0])
        self.assertTrue(report["parameters_changed"])
        self.assertTrue(report["sampling_log_prob_requires_grad"])
        self.assertTrue(any(torch.tensor(g).abs().sum() > 0 for g in report["policy_gradients"].values()))
        self.assertEqual(report["real_api_calls"], 0)
        json.dumps(report)

    def test_zero_reward_policy_only_has_no_update(self):
        policy = TopologyPolicy(FULL, FULL)
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.1)
        report = offline_training_step(policy, optimizer, self.config,
            [{"sample_id": "synthetic:wrong", "task": "3 + 2?", "target": "6"}],
            generator=torch.Generator().manual_seed(0), include_regularization=False)
        self.assertEqual(report["rewards"], [0])
        self.assertFalse(report["parameters_changed"])

    def test_invalid_settings_and_cycles(self):
        for probability in [0, 1, float("nan")]:
            with self.assertRaises(ValueError):
                TopologyPolicy(FULL, FULL, probability)
        with self.assertRaises(ValueError):
            execution_topology([["A", "B"], ["B", "A"]], [])
        with self.assertRaises(ValueError):
            build_fixed_graph(self.config, round_topologies=[execution_topology([], [["A", "A"]])] * 2)

    @unittest.skipUnless(REFERENCE.exists(), "原项目仅用于迁移对照")
    def test_sample_edges_log_probs_and_gradients_match_original(self):
        namespace = original_namespace()
        class Node:
            def __init__(self, name):
                self.id = name
                self.spatial_successors, self.spatial_predecessors = [], []
                self.temporal_successors, self.temporal_predecessors = [], []
            def add_successor(self, other, kind):
                getattr(self, kind + "_successors").append(other)
                getattr(other, kind + "_predecessors").append(self)
        for seed in [0, 1, 4, 12]:
            policy = TopologyPolicy(FULL, FULL, 0.4)
            graph = namespace["Graph"].__new__(namespace["Graph"])
            graph.nodes = {node: Node(node) for node in ("A", "B")}
            graph.decision_node = Node("final")
            graph.potential_spatial_edges = [(src, dst) for src in graph.nodes for dst in graph.nodes]
            graph.potential_temporal_edges = graph.potential_spatial_edges
            graph.spatial_masks = graph.temporal_masks = torch.ones(4)
            graph.spatial_logits = policy.spatial_logits.detach().flatten().clone().requires_grad_()
            graph.temporal_logits = policy.temporal_logits.detach().flatten().clone().requires_grad_()
            graph.optimized_spatial = graph.optimized_temporal = True
            torch.manual_seed(seed)
            original_logs = []
            original_edges = []
            for round_index in range(2):
                original_logs.extend([graph.construct_spatial_connection(temperature=2),
                                      graph.construct_temporal_connection(round_index, temperature=2)])
                original_edges.append({kind: tuple((node.id, other.id) for node in graph.nodes.values()
                    for other in getattr(node, kind + "_successors")) for kind in ("spatial", "temporal")})
            sampled = policy.sample_rounds(temperature=2, generator=torch.Generator().manual_seed(seed))
            for item, edges in zip(sampled, original_edges):
                self.assertEqual(item.spatial_edges, edges["spatial"])
                self.assertEqual(item.temporal_edges, edges["temporal"])
            total, original_total = sum(item.log_prob for item in sampled), sum(original_logs)
            torch.testing.assert_close(total, original_total)
            grads = torch.autograd.grad(total, tuple(policy.parameters()))
            original_grads = torch.autograd.grad(original_total, (graph.spatial_logits, graph.temporal_logits))
            for actual, expected in zip(grads, original_grads):
                torch.testing.assert_close(actual, expected.reshape(2, 2))

    @unittest.skipUnless(REFERENCE.exists(), "原项目仅用于迁移对照")
    def test_regularizer_matches_original_values_and_gradients(self):
        import ast
        namespace = {"torch": torch}
        execute_nodes([node for node in parse("AgentDropout/utils/utils.py").body
                       if isinstance(node, ast.FunctionDef) and node.name in {"nuclear_norm", "frobenius_norm"}], namespace)
        policy = TopologyPolicy(((0, 1), (1, 0)), FULL, 0.4)
        old = [p.detach().clone().requires_grad_() for p in policy.parameters()]
        original = sum(namespace["nuclear_norm"](p) + torch.relu(namespace["frobenius_norm"](mask, p) - 0.1)
                       for p, mask in zip(old, (policy.spatial_masks, policy.temporal_masks)))
        new = policy.regularization_loss()
        torch.testing.assert_close(new, original)
        for actual, expected in zip(torch.autograd.grad(new, tuple(policy.parameters())), torch.autograd.grad(original, old)):
            torch.testing.assert_close(actual, expected)

    def test_cli_saves_checkpoint_without_live_log_probs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "step"
            # 直接执行 CLI 入口；避免在已加载 OpenMP 的测试进程中再初始化子进程运行时。
            with patch.object(sys, "argv", ["train.py", "--output-dir", str(output)]), redirect_stdout(io.StringIO()):
                runpy.run_path(str(ROOT / "train.py"), run_name="__main__")
            report = json.loads((output / "training.json").read_text())
            checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
            self.assertTrue(report["parameters_changed"])
            self.assertEqual(report["real_api_calls"], 0)
            self.assertNotIn("log_prob", checkpoint)
            self.assertTrue(all(not value.requires_grad for value in checkpoint["model_state_dict"].values()))
            restored = TopologyPolicy(FULL, FULL)
            restored.load_state_dict(checkpoint["model_state_dict"])
            self.assertEqual(restored.spatial_logits.detach().tolist(), report["after"]["spatial_logits"])
            rng = torch.Generator().set_state(checkpoint["generator_state"])
            first = [item.for_report() for item in restored.sample_rounds(generator=rng)]
            rng.set_state(checkpoint["generator_state"])
            second = [item.for_report() for item in restored.sample_rounds(generator=rng)]
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
