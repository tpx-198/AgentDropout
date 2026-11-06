import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from ad_l.fixed_graph import build_fixed_graph, load_fixed_graph_config
from ad_l.node_pruning import prune_nodes, sample_skips
from ad_l.topology import TopologyPolicy
from original_fixed_graph_reference import REFERENCE, original_namespace


class ThreeAgentTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.config = load_fixed_graph_config(root / "configs/three_agents_offline.json")

    @unittest.skipUnless(REFERENCE.exists(), "原项目仅用于提示词对照")
    def test_inspector_uses_original_prompt_with_round_visibility(self):
        state = build_fixed_graph(self.config).invoke({"task": "3 + 2?"})
        self.assertEqual(state["model_calls"], 7)
        self.assertEqual(list(state["trace"][-1]["spatial_info"]), ["A", "B", "C"])
        original = original_namespace()
        prompt_set = original["GSM8KPromptSet"]
        inspector = SimpleNamespace(role="Inspector", prompt_set=prompt_set,
                                    constraint=prompt_set.get_constraint("Inspector"))
        for call in (entry for entry in state["trace"] if entry["node_id"] == "C"):
            self.assertEqual(list(call["spatial_info"]), ["A"])
            self.assertEqual(list(call["temporal_info"]), [] if call["round"] == 1 else ["A", "B"])
            system, user = original["math_prompt"](inspector, {"task": "3 + 2?"},
                                                   call["spatial_info"], call["temporal_info"])
            self.assertEqual(call["messages"], [{"role": "system", "content": system},
                                               {"role": "human", "content": user}])

    def test_asymmetric_selection_and_surviving_edge_sampling(self):
        config = self.config
        selector = TopologyPolicy(config.spatial_masks, config.temporal_masks, 0.3,
                                  diff=True, node_ids=config.node_ids)
        generator = torch.Generator().manual_seed(0)
        _, probabilities, surrogate = sample_skips(selector.spatial_logits, selector.fixed_round_topologies(config),
                                                   generator, node_ids=config.node_ids)
        self.assertLess(probabilities[0][0], probabilities[0][1])
        self.assertEqual(probabilities[0][1], probabilities[0][2])
        surrogate.backward()
        self.assertNotEqual(selector.spatial_logits.grad[:, 0, 2].abs().sum().item(), 0)

        _, after, plan, _ = prune_nodes(selector, config)
        self.assertEqual(plan, ("B", "B"))
        self.assertEqual(after.spatial_masks[:, 0, 2].tolist(), [1, 1])
        sampled = after.sample_rounds(generator=generator)
        for round_data in sampled:
            self.assertIn("C", round_data.for_execution()["execution_order"])
            self.assertTrue(all("B" not in (edge["source"], edge["target"]) for edge in round_data.decisions))
        state = build_fixed_graph(config, round_topologies=[item.for_execution() for item in sampled],
                                  skip_nodes=plan).invoke({"task": "3 + 2?"})
        self.assertEqual([call["node_id"] for call in state["trace"]], ["A", "C", "A", "C", "final"])
        self.assertEqual(state["trace"][-1]["spatial_info"]["B"]["output"], "None.")
