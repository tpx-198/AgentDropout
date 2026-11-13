import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from ad_l.topology import EdgePolicy, TOPOLOGIES, TopologyConfig, load_policy, save_policy, topological_order


class TopologyTests(unittest.TestCase):
    def test_all_topologies_sizes_rounds_and_dag_sampling(self):
        for agents in (2, 5, 9):
            for rounds in (1, 3):
                for name in TOPOLOGIES:
                    config = TopologyConfig(agents=agents, rounds=rounds, topology=name)
                    policy = EdgePolicy(config)
                    for mode in ("sampled", "fixed", "threshold"):
                        result = policy.sample(torch.Generator().manual_seed(7), mode=mode)
                        self.assertEqual(len(result.execution["rounds"]), rounds)
                        self.assertEqual(result.execution["rounds"][0]["temporal"], [])
                        for graph in result.execution["rounds"]:
                            self.assertEqual(len(topological_order(config.workers, graph["spatial"])), config.workers)
                        json.dumps(result.execution)
                    for weights in policy.weights().values():
                        self.assertTrue(((0 <= weights) & (weights <= 1)).all())

    def test_random_candidates_and_samples_are_reproducible(self):
        config = TopologyConfig(topology="random", rounds=3, seed=11)
        first, second = EdgePolicy(config), EdgePolicy(config)
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name])
        sample1 = first.sample(torch.Generator().manual_seed(3))
        sample2 = second.sample(torch.Generator().manual_seed(3))
        self.assertEqual(sample1.execution, sample2.execution)
        torch.testing.assert_close(sample1.log_prob, sample2.log_prob)
        other = EdgePolicy(replace(config, seed=12))
        self.assertFalse(torch.equal(first.spatial_mask, other.spatial_mask))

    def test_gradient_matches_bernoulli_formula(self):
        policy = EdgePolicy(TopologyConfig(topology="chain", temporal="upper", rounds=3))
        sample = policy.sample(torch.Generator().manual_seed(4))
        self.assertTrue(sample.log_prob.requires_grad)
        sample.log_prob.backward()
        for round_index, graph in enumerate(sample.execution["rounds"]):
            expected = torch.zeros_like(policy.spatial_logits[round_index])
            for source, target in torch.nonzero(policy.spatial_mask[round_index]).tolist():
                expected[source, target] = float([source, target] in graph["spatial"]) - 0.5
            torch.testing.assert_close(policy.spatial_logits.grad[round_index], expected)
            if round_index:
                expected = torch.zeros_like(policy.temporal_logits[round_index - 1])
                for source, target in torch.nonzero(policy.temporal_mask[round_index - 1]).tolist():
                    expected[source, target] = float([source, target] in graph["temporal"]) - 0.5
                torch.testing.assert_close(policy.temporal_logits.grad[round_index - 1], expected)
        expected_summary = torch.tensor([float(i in sample.execution["summary"]) - 0.5 for i in range(4)])
        torch.testing.assert_close(policy.summary_logits.grad, expected_summary)

    def test_pruning_is_per_matrix_monotonic_and_zero_is_noop(self):
        policy = EdgePolicy(TopologyConfig())
        before = {name: tensor.clone() for name, tensor in policy.state_dict().items()}
        policy.prune(0)
        for name, tensor in before.items():
            torch.testing.assert_close(tensor, policy.state_dict()[name])
        with torch.no_grad():
            policy.summary_logits.copy_(torch.tensor([-2.0, 2.0, -1.0, 1.0]))
        report = policy.prune(0.25)
        self.assertEqual(policy.spatial_mask.sum((1, 2)).tolist(), [9, 9])
        self.assertEqual(policy.summary_mask.tolist(), [False, True, True, True])
        self.assertEqual(report["temporal"], {"before": 6, "after": 5})
        self.assertEqual(policy.weights()["summary"][0], 0)
        policy.prune(1)
        sampled = policy.sample(torch.Generator().manual_seed(0))
        sampled.log_prob.backward()
        self.assertEqual(sampled.execution["summary"], [])
        for graph in sampled.execution["rounds"]:
            self.assertEqual(graph["spatial"], [])
            self.assertEqual(graph["temporal"], [])
        for parameter in policy.parameters():
            self.assertTrue((parameter.grad == 0).all())

    def test_checkpoint_and_corruption_validation(self):
        policy = EdgePolicy(TopologyConfig(topology="layered"))
        policy.prune(0.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            save_policy(path, policy, {"status": "test"})
            loaded, metadata = load_policy(path)
            self.assertEqual(metadata["status"], "test")
            self.assertEqual(loaded.report(), policy.report())
            payload = torch.load(path, weights_only=True)
            payload["state"]["spatial_mask"][0, 0, 0] = True
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "outside"):
                load_policy(path)

    def test_invalid_arguments_and_cycles(self):
        for settings in ({"agents": 1}, {"rounds": 0}, {"initial_weight": 1}, {"initial_weight": float('nan')},
                         {"density": -0.1}, {"layers": 0}, {"topology": "missing"}):
            with self.assertRaises(ValueError):
                TopologyConfig(**settings)
        for edges in ([[0, 1], [1, 0]], [[0, 0]], [[0, 1], [0, 1]], [[0, 4]]):
            with self.assertRaises(ValueError):
                topological_order(4, edges)
        policy = EdgePolicy(TopologyConfig())
        for rate in (-0.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                policy.prune(rate)


if __name__ == "__main__":
    unittest.main()
