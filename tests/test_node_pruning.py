import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import torch

from ad_l.datasets import GSM8KSample
from ad_l.fixed_graph import build_fixed_graph, load_fixed_graph_config
from ad_l.node_pruning import load_node_pruning, prune_nodes, sample_skips, train_selector_step
from ad_l.topology import TopologyPolicy
from ad_l.training import policy_from_checkpoint


class NodePruningTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.config = load_fixed_graph_config(root / "configs/fixed_graph_offline.json")
        self.selector = TopologyPolicy(self.config.spatial_masks, self.config.temporal_masks, diff=True)

    def test_original_skip_surrogate_and_selector_update(self):
        topologies = self.selector.fixed_round_topologies(self.config)
        with torch.no_grad():
            self.selector.spatial_logits[0, 0, 1] = 0.7
            self.selector.spatial_logits[1, 0, 1] = -0.4
        plan, probabilities, surrogate = sample_skips(
            self.selector.spatial_logits, topologies, torch.Generator().manual_seed(0))
        self.assertEqual(len(plan), 2)
        self.assertEqual(probabilities, [[0.5, 0.5], [0.5, 0.5]])
        logits = torch.tensor([0.7, -0.4], requires_grad=True)
        p = logits.sigmoid()
        expected = (4 * (1 - p).log() + p.log()).sum()
        torch.testing.assert_close(surrogate, expected)
        grad, = torch.autograd.grad(surrogate, self.selector.spatial_logits)
        expected_grad, = torch.autograd.grad(expected, logits)
        torch.testing.assert_close(grad[:, 0, 1], expected_grad)
        self.assertEqual(grad.count_nonzero().item(), 2)

        temporal = self.selector.temporal_logits.detach().clone()
        optimizer = torch.optim.Adam(self.selector.parameters(), lr=0.1)
        samples = [GSM8KSample("train:0", "train", 0, "3+2?", "5")]
        report = train_selector_step(self.selector, optimizer, self.config, samples,
                                     generator=torch.Generator().manual_seed(0))
        self.assertTrue(report["parameters_changed"])
        self.assertEqual(report["episodes"][0]["model_calls"], 3)
        torch.testing.assert_close(self.selector.temporal_logits, temporal)

    def test_skipped_output_is_visible_on_unpruned_graph_in_sync_and_async(self):
        graph = build_fixed_graph(self.config, skip_nodes=("A", "B"))
        for state in (graph.invoke({"task": "3+2?"}), asyncio.run(graph.ainvoke({"task": "3+2?"}))):
            self.assertEqual([call["node_id"] for call in state["trace"]], ["B", "A", "final"])
            self.assertEqual(state["trace"][0]["spatial_info"]["A"]["output"], "None.")
            self.assertEqual(state["trace"][-1]["spatial_info"]["B"]["output"], "None.")
            self.assertEqual(state["model_calls"], 3)
            self.assertEqual(state["real_api_calls"], 0)

    def test_pruning_disconnects_different_nodes_in_each_round(self):
        with torch.no_grad():
            self.selector.spatial_logits[0, 0, 0] = -1
            self.selector.spatial_logits[1, 1, 1] = -1
        before, after, plan, report = prune_nodes(self.selector, self.config)
        self.assertEqual(plan, ("A", "B"))
        self.assertEqual(report["temporal_masks"], [[[0, 0], [1, 0]]])
        self.assertEqual(report["effective_before"][0]["spatial_edges"], [["A", "B"]])
        self.assertEqual(report["effective_after"][1]["temporal_edges"], [["B", "A"]])
        state = build_fixed_graph(self.config, round_topologies=after.fixed_round_topologies(self.config),
                                  skip_nodes=plan).invoke({"task": "3+2?"})
        self.assertEqual(state["trace"][0]["spatial_info"], {})
        self.assertEqual(state["trace"][1]["temporal_info"]["B"]["output"], self.config.responses[1][0])
        self.assertEqual(state["trace"][-1]["spatial_info"]["B"]["output"], "None.")
        self.assertEqual(before.temporal_masks.sum().item(), 4)

    def test_edge_training_imports_legacy_report_and_rejects_reenabled_edges(self):
        _, after, _, report = prune_nodes(self.selector, self.config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pruning.json"
            report.pop("source_config")
            path.write_text(json.dumps(report))
            path.with_name("manifest.json").write_text(json.dumps({"config": self.config.to_dict()}))
            imported = load_node_pruning(path, self.config)
            report["temporal_masks"][0][0][0] = 1
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "掩码与跳过计划不一致"):
                load_node_pruning(path, self.config)
        checkpoint = {"context": {"diff": True, "node_pruning": imported, "config": self.config.to_dict()},
                      "model_state_dict": after.state_dict()}
        policy = policy_from_checkpoint(checkpoint)
        self.assertEqual(policy.temporal_masks.sum().item(), 1)
        checkpoint["model_state_dict"]["temporal_masks"][0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "重新启用"):
            policy_from_checkpoint(checkpoint)
