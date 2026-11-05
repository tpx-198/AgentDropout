"""逐轮独立参数的原实现对照、恢复、冻结评测和入口验证。"""

import ast
import copy
import io
import json
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

from ad_l.datasets import GSM8KSample
from ad_l.evaluation import run_baselines
from ad_l.fixed_graph import build_fixed_graph, load_fixed_graph_config
from ad_l.models import ModelSettings
from ad_l.pruning import fixed_execution, prune_checkpoint, prune_edges, verify_pair
from ad_l.topology import TopologyPolicy
from ad_l.training import load_training_checkpoint, policy_from_checkpoint, train_multistep
from original_fixed_graph_reference import REFERENCE, execute_nodes, original_namespace, parse

ROOT = Path(__file__).resolve().parents[1]
FULL = ((1, 1), (1, 1))


def reference_graph(policy):
    namespace = original_namespace()
    graph = namespace["Graph"].__new__(namespace["Graph"])

    class Node:
        def __init__(self, node):
            self.id = node
            self.spatial_successors, self.spatial_predecessors = [], []
            self.temporal_successors, self.temporal_predecessors = [], []

        def add_successor(self, other, kind):
            getattr(self, kind + "_successors").append(other)
            getattr(other, kind + "_predecessors").append(self)

    graph.nodes = {node: Node(node) for node in ("A", "B")}
    graph.decision_node = Node("final")
    graph.potential_spatial_edges = [(src, dst) for src in graph.nodes for dst in graph.nodes]
    graph.potential_temporal_edges = graph.potential_spatial_edges
    graph.optimized_spatial = graph.optimized_temporal = True
    graph.diff, graph.rounds = True, 2
    for kind in ("spatial", "temporal"):
        setattr(graph, kind + "_masks", [item.clone().flatten() for item in getattr(policy, kind + "_masks")])
        setattr(graph, kind + "_logits", [item.detach().clone().flatten().requires_grad_()
                                          for item in getattr(policy, kind + "_logits")])
    return graph


class RoundTopologyTests(unittest.TestCase):
    def setUp(self):
        self.network = patch("socket.socket.connect", side_effect=AssertionError("offline test"))
        self.network.start()
        self.quiet = redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.config = load_fixed_graph_config(ROOT / "configs/fixed_graph_offline.json")
        self.settings = ModelSettings()
        self.pool = [GSM8KSample(f"gsm8k:train:{i}", "train", i, "3 + 2?", "5") for i in range(3)]
        self.context = {"dataset": {"split": "train"}, "diff": True,
            "pool_sample_ids": [sample.sample_id for sample in self.pool], "settings": asdict(self.settings),
            "config": self.config.to_dict(), "hyperparameters": {"seed": 3, "batch_size": 1,
                "lr": 0.1, "initial_probability": 0.4, "temperature": 1.0, "delta": 0.1,
                "prune_every": 2, "prune_rate": 0.25, "prune_until": 4}}

    def tearDown(self):
        self.quiet.__exit__(None, None, None)
        self.network.stop()

    def test_round_gradient_isolation_and_validation(self):
        policy = TopologyPolicy(FULL, FULL, diff=True)
        self.assertEqual(tuple(policy.spatial_logits.shape), (2, 2, 2))
        self.assertEqual(tuple(policy.temporal_logits.shape), (1, 2, 2))
        first, second = policy.sample_rounds(generator=torch.Generator().manual_seed(4))
        grad_s, grad_t = torch.autograd.grad(first.log_prob, tuple(policy.parameters()), retain_graph=True)
        self.assertGreater(grad_s[0].abs().sum().item(), 0)
        self.assertEqual(grad_s[1].abs().sum().item(), 0)
        self.assertEqual(grad_t.abs().sum().item(), 0)
        grad_s, grad_t = torch.autograd.grad(second.log_prob, tuple(policy.parameters()))
        self.assertEqual(grad_s[0].abs().sum().item(), 0)
        self.assertGreater(grad_s[1].abs().sum().item(), 0)
        self.assertGreater(grad_t.abs().sum().item(), 0)
        for rounds in (1, 3):
            with self.assertRaises(ValueError):
                policy.sample_rounds(rounds)
        with self.assertRaises(ValueError):
            TopologyPolicy(FULL, FULL, diff="true")

    @unittest.skipUnless(REFERENCE.exists(), "原项目仅用于迁移对照")
    def test_sampler_matches_original_edges_values_and_gradients(self):
        for seed in (0, 1, 4, 12):
            policy = TopologyPolicy(FULL, FULL, diff=True)
            with torch.no_grad():
                policy.spatial_logits.copy_(torch.tensor([[[0.1, -0.7], [0.4, 1.2]], [[0.3, 0.8], [-0.2, -0.4]]]))
                policy.temporal_logits.copy_(torch.tensor([[[0.3, -0.6], [1.1, -0.2]]]))
            graph = reference_graph(policy)
            torch.manual_seed(seed)
            logs, edges = [], []
            for index in range(2):
                logs.extend([graph.construct_spatial_connection_diff(index, temperature=2),
                             graph.construct_temporal_connection_diff(index, temperature=2)])
                edges.append({kind: tuple((node.id, dst.id) for node in graph.nodes.values()
                    for dst in getattr(node, kind + "_successors")) for kind in ("spatial", "temporal")})
            sampled = policy.sample_rounds(temperature=2, generator=torch.Generator().manual_seed(seed))
            for item, original in zip(sampled, edges):
                self.assertEqual(item.spatial_edges, original["spatial"])
                self.assertEqual(item.temporal_edges, original["temporal"])
            total, old_total = sum(item.log_prob for item in sampled), sum(logs)
            torch.testing.assert_close(total, old_total)
            expected = torch.autograd.grad(old_total, (*graph.spatial_logits, *graph.temporal_logits))
            actual = torch.autograd.grad(total, tuple(policy.parameters()))
            torch.testing.assert_close(actual[0], torch.stack([item.reshape(2, 2) for item in expected[:2]]))
            torch.testing.assert_close(actual[1], expected[2].reshape(1, 2, 2))

    @unittest.skipUnless(REFERENCE.exists(), "原项目仅用于迁移对照")
    def test_regularizer_matches_original_means_and_initial_reference_after_pruning(self):
        namespace = {"torch": torch}
        execute_nodes([node for node in parse("AgentDropout/utils/utils.py").body
                       if isinstance(node, ast.FunctionDef) and node.name in {"nuclear_norm", "frobenius_norm"}], namespace)
        policy = TopologyPolicy(FULL, FULL, diff=True)
        with torch.no_grad():
            policy.spatial_logits.copy_(torch.tensor([[[0.4, 0.2], [0.1, 0.7]], [[1.3, 0.8], [0.2, 0.1]]]))
            policy.temporal_logits.copy_(torch.tensor([[[0.4, 0.8], [0.1, 1.2]]]))
        previous = policy.regularization_loss(1.32, reference_masks=(FULL, FULL))
        prune_edges(policy, 0.5)
        new = policy.regularization_loss(1.32, reference_masks=(FULL, FULL))
        torch.testing.assert_close(new, previous)
        old = [p.detach().clone().requires_grad_() for p in policy.parameters()]
        original = sum(torch.stack([namespace["nuclear_norm"](p) for p in group]).mean()
                       + torch.relu(torch.stack([namespace["frobenius_norm"](torch.tensor(FULL), p)
                                                 for p in group]).mean() - 1.32) for group in old)
        torch.testing.assert_close(new, original)
        for actual, expected in zip(torch.autograd.grad(new, tuple(policy.parameters())), torch.autograd.grad(original, old)):
            torch.testing.assert_close(actual, expected)

    @unittest.skipUnless(REFERENCE.exists(), "原项目仅用于迁移对照")
    def test_pruning_sorts_each_matrix_like_original(self):
        for rate in (0, 0.25, 0.5, 1):
            policy = TopologyPolicy(FULL, FULL, diff=True)
            with torch.no_grad():
                policy.spatial_logits[1].copy_(torch.tensor([[3., 1.], [2., 0.]]))
                policy.spatial_masks[0, 0, 0] = 0
            graph = reference_graph(policy)
            graph.update_masks_diff(rate)
            report = prune_edges(policy, rate)
            for kind in ("spatial", "temporal"):
                torch.testing.assert_close(getattr(policy, kind + "_masks"),
                    torch.stack([mask.reshape(2, 2) for mask in getattr(graph, kind + "_masks")]))
                self.assertEqual(len(report["changes"][kind]["matrices"]), 2 if kind == "spatial" else 1)
            self.assertEqual(report["rule"], "agentdropout_update_masks_diff_v1")

    @unittest.skipUnless(REFERENCE.exists(), "原项目仅用于迁移对照")
    def test_fixed_round_connections_and_prompt_visibility(self):
        policy = TopologyPolicy(FULL, FULL, diff=True)
        with torch.no_grad():
            policy.spatial_masks.copy_(torch.tensor([[[0, 1], [0, 0]], [[0, 0], [1, 0]]]))
        topologies = fixed_execution(policy, self.config)
        graph = reference_graph(policy)
        graph.optimized_spatial = graph.optimized_temporal = False
        for index in range(2):
            graph.construct_spatial_connection_diff(index)
            graph.construct_temporal_connection_diff(index)
            for kind in ("spatial", "temporal"):
                edges = [[node.id, dst.id] for node in graph.nodes.values()
                         for dst in getattr(node, kind + "_successors")]
                self.assertEqual(topologies[index][kind + "_edges"], edges)
        state = build_fixed_graph(self.config, round_topologies=topologies).invoke({"task": "3+2?"})
        self.assertEqual([item["node_id"] for item in state["trace"]], ["A", "B", "B", "A", "final"])
        self.assertEqual(set(state["trace"][3]["spatial_info"]), {"B"})
        self.assertEqual(set(state["trace"][3]["temporal_info"]), {"B"})
        self.assertEqual(state["model_calls"], 5)

    def test_resume_with_cyclic_pruning_is_exact_and_checkpoint_mode_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def train(name, steps, checkpoint=None):
                train_multistep(self.pool, self.config, self.settings, self.context, root / name, steps, checkpoint=checkpoint)
                return load_training_checkpoint(root / name / "checkpoint.pt")
            full = train("full", 5)
            first = train("first", 2)
            resumed = train("resumed", 5, first)
            for name, value in full["model_state_dict"].items():
                torch.testing.assert_close(value, resumed["model_state_dict"][name], rtol=0, atol=0)
            for index, values in full["optimizer_state_dict"]["state"].items():
                for name, value in values.items():
                    torch.testing.assert_close(value, resumed["optimizer_state_dict"]["state"][index][name], rtol=0, atol=0)
            self.assertTrue(torch.equal(full["generator_state"], resumed["generator_state"]))
            self.assertEqual(full["pruning_state"], resumed["pruning_state"])
            self.assertEqual(full["cursor"], resumed["cursor"])
            self.assertEqual([event["step"] for event in full["pruning_state"]["events"]], [2, 4])
            self.assertEqual(first["pruning_state"]["events"][0]["changes"]["spatial"]["actual_removed"], 2)
            corrupt = copy.deepcopy(full)
            corrupt["context"]["diff"] = False
            with self.assertRaisesRegex(ValueError, "形状"):
                policy_from_checkpoint(corrupt)

    def test_shared_training_keeps_initial_regularization_reference_after_pruning(self):
        self.context["diff"] = False
        self.context["hyperparameters"]["prune_every"] = 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "train"
            train_multistep(self.pool, self.config, self.settings, self.context, root, 2)
            history = [json.loads(line) for line in (root / "history.jsonl").read_text().splitlines()]
            report = history[1]
            policy = TopologyPolicy(history[0]["pruning_event"]["changes"]["spatial"]["after_mask"],
                                    history[0]["pruning_event"]["changes"]["temporal"]["after_mask"])
            with torch.no_grad():
                for name, value in report["before"].items():
                    getattr(policy, name).copy_(torch.tensor(value))
            expected = policy.regularization_loss(reference_masks=(self.config.spatial_masks, self.config.temporal_masks))
            self.assertAlmostEqual(report["regularization_loss"], expected.item())
            self.assertNotAlmostEqual(report["regularization_loss"], policy.regularization_loss().item())
            self.assertEqual(report["regularization_reference"], "initial_config_masks")

    def test_frozen_sampled_and_fixed_evaluation_and_pruned_provenance(self):
        policy = TopologyPolicy(FULL, FULL, diff=True)
        checkpoint = {"step": 1, "context": {"diff": True}, "model_state_dict": copy.deepcopy(policy.state_dict())}
        pruned, report, before, after = prune_checkpoint(checkpoint, self.config, 0.25, parent_sha256="fixture")
        verify_pair(checkpoint, pruned, "fixture")
        self.assertTrue(policy_from_checkpoint(pruned).diff)
        self.assertEqual(len(report["effective_before"]), 2)
        samples = [GSM8KSample("gsm8k:test:0", "test", 0, "3+2?", "5")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("sampled", "original-fixed"):
                result = run_baselines(samples, self.config, self.settings, root / mode, {}, baselines=("learned",),
                    learned_policy=policy, learned_generator=torch.Generator().manual_seed(3), learned_mode=mode)
                self.assertEqual(result["status"], "success")
                row = json.loads((root / mode / "predictions.jsonl").read_text())
                self.assertEqual(row["model_calls"], 5)
                self.assertEqual(row["real_api_calls"], 0)
                self.assertEqual(len(row["sampled_rounds"] if mode == "sampled" else row["effective_topology"]), 2)
            result = run_baselines(samples, self.config, self.settings, root / "pair", {}, baselines=("before", "after"),
                learned_mode="original-fixed", policy_variants={"before": before, "after": after})
            self.assertEqual(result["status"], "success")
        for name, value in policy.state_dict().items():
            torch.testing.assert_close(value, checkpoint["model_state_dict"][name], rtol=0, atol=0)

    @unittest.skipUnless((ROOT / "data/gsm8k/train.jsonl").exists(), "官方数据需先准备")
    def test_cli_diff_plan_resume_inheritance_and_mode_switch_rejection(self):
        import scripts.train_topology as entry
        stdout = io.StringIO()
        with patch.object(sys, "argv", ["train.py", "--diff", "--steps", "4", "--prune-every", "2", "--plan", "--deepseek"]), \
                patch.object(entry, "create_model_adapter") as factory, patch("dotenv.load_dotenv") as load, redirect_stdout(stdout):
            runpy.run_path(str(ROOT / "train.py"), run_name="__main__")
        self.assertTrue(json.loads(stdout.getvalue())["diff"])
        factory.assert_not_called()
        load.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for argv in (["--diff", "--steps", "1", "--output-dir", str(root / "first")],
                         ["--resume", str(root / "first/checkpoint.pt"), "--steps", "2", "--output-dir", str(root / "resumed")]):
                with patch.object(sys, "argv", ["train.py", *argv]):
                    runpy.run_path(str(ROOT / "train.py"), run_name="__main__")
            self.assertTrue(policy_from_checkpoint(load_training_checkpoint(root / "resumed/checkpoint.pt")).diff)
            with patch.object(sys, "argv", ["train.py", "--resume", str(root / "first/checkpoint.pt"), "--no-diff", "--plan"]), \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runpy.run_path(str(ROOT / "train.py"), run_name="__main__")


if __name__ == "__main__":
    unittest.main()
