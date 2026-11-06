"""验证原剪枝规则、来源不变及统一推理的配对评测。"""

import copy
import hashlib
import io
import json
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from ad_l.datasets import GSM8KSample
from ad_l.evaluation import run_baselines
from ad_l.fixed_graph import load_fixed_graph_config
from ad_l.models import ModelAdapter, ModelSettings
from ad_l.pruning import compare_pairs, prune_checkpoint, prune_edges, verify_pair
from ad_l.topology import TopologyPolicy
from ad_l.training import load_training_checkpoint, train_multistep
from original_fixed_graph_reference import REFERENCE, original_namespace

ROOT = Path(__file__).resolve().parents[1]


class PruningTests(unittest.TestCase):
    def setUp(self):
        self.network = patch("socket.socket.connect", side_effect=AssertionError("offline test"))
        self.network.start()
        self.quiet = redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.config = load_fixed_graph_config(ROOT / "configs/fixed_graph_offline.json")

    def tearDown(self):
        self.quiet.__exit__(None, None, None)
        self.network.stop()

    def checkpoint(self):
        policy = TopologyPolicy(self.config.spatial_masks, self.config.temporal_masks)
        return {"schema_version": 2, "step": 1, "cursor": 2,
                "model_state_dict": policy.state_dict(), "optimizer_state_dict": {},
                "generator_state": torch.Generator().get_state(),
                "context": {"dataset": {"split": "train"}, "settings": asdict(ModelSettings()),
                            "config": self.config.to_dict(), "hyperparameters": {"batch_size": 2}}}

    def test_masks_only_minimum_one_empty_edges_and_invalid_rate(self):
        policy = TopologyPolicy(self.config.spatial_masks, self.config.temporal_masks)
        before = {name: value.clone() for name, value in policy.named_parameters()}
        record = prune_edges(policy, 0)
        self.assertEqual(record["changes"]["spatial"]["actual_removed"], 1)
        self.assertEqual(record["changes"]["temporal"]["actual_removed"], 1)
        for name, parameter in policy.named_parameters():
            torch.testing.assert_close(parameter, before[name])
        prune_edges(policy, 1)
        empty = prune_edges(policy, 0.25)
        self.assertEqual(empty["changes"]["spatial"]["actual_removed"], 0)
        self.assertEqual(empty["changes"]["temporal"]["actual_removed"], 0)
        for rate in (-1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                prune_edges(policy, rate)

    @unittest.skipUnless(REFERENCE.exists(), "原项目仅用于迁移对照")
    def test_masks_match_original_update_masks_including_ties(self):
        namespace = original_namespace()
        for masks in [((0, 1), (0, 0)), ((1, 1), (1, 1)), ((0, 0), (0, 0))]:
            for rate in (0, 0.25, 0.5, 1):
                for weights in (torch.zeros(2, 2), torch.tensor([[0.3, -0.2], [0.7, -0.1]])):
                    policy = TopologyPolicy(masks, masks)
                    with torch.no_grad():
                        policy.spatial_logits.copy_(weights)
                        policy.temporal_logits.copy_(weights)
                    graph = namespace["Graph"].__new__(namespace["Graph"])
                    graph.optimized_spatial = graph.optimized_temporal = True
                    graph.spatial_logits = graph.temporal_logits = weights.flatten().clone()
                    graph.spatial_masks = torch.tensor(masks).flatten().clone()
                    graph.temporal_masks = graph.spatial_masks.clone()
                    expected = graph.update_masks(rate)
                    prune_edges(policy, rate)
                    torch.testing.assert_close(policy.spatial_masks, expected[0].reshape(2, 2).float())
                    torch.testing.assert_close(policy.temporal_masks, expected[1].reshape(2, 2).float())

    def test_checkpoint_parent_is_unchanged_and_pair_is_verified(self):
        original = self.checkpoint()
        saved = copy.deepcopy(original)
        pruned, record, before, after = prune_checkpoint(original, self.config, 0.25, parent_sha256="parent")
        verify_pair(original, pruned, "parent")
        self.assertTrue(pruned["evaluation_only"])
        for name in saved["model_state_dict"]:
            torch.testing.assert_close(original["model_state_dict"][name], saved["model_state_dict"][name])
        self.assertEqual(pruned["context"], original["context"])
        with self.assertRaises(ValueError):
            verify_pair(original, pruned, "wrong-parent")
        pruned["model_state_dict"]["spatial_logits"][0, 1] += 1
        with self.assertRaises(ValueError):
            verify_pair(original, pruned, "parent")

    def test_fixed_temporal_filter_can_activate_edges_after_spatial_pruning(self):
        checkpoint = self.checkpoint()
        checkpoint["model_state_dict"]["temporal_logits"][0, 0] = -1
        pruned, record, _, _ = prune_checkpoint(checkpoint, self.config, 0.25, parent_sha256="parent")
        self.assertIn(["B", "A"], record["changes"]["temporal"]["effective_added"])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "仅供评测"):
                train_multistep([GSM8KSample("train", "train", 0, "Q", "5")], self.config, ModelSettings(),
                                checkpoint["context"], Path(directory) / "bad-resume", 2, checkpoint=pruned)

    def test_pair_records_use_same_samples_mode_and_unknown_tokens(self):
        _, pruning, before, after = prune_checkpoint(self.checkpoint(), self.config, 0.25, parent_sha256="parent")
        samples = [GSM8KSample("gsm8k:test:0", "test", 0, "3 + 2?", "5")]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "eval"
            report = run_baselines(samples, self.config, ModelSettings(), output, {}, baselines=("before", "after"),
                learned_mode="original-fixed", policy_variants={"before": before, "after": after})
            rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual([row["sample_id"] for row in rows], [samples[0].sample_id] * 2)
            self.assertEqual([row["learned_mode"] for row in rows], ["original-fixed"] * 2)
            self.assertEqual(rows[0]["effective_topology"], pruning["effective_before"])
            self.assertEqual(rows[1]["effective_topology"], pruning["effective_after"])
            comparison = compare_pairs(rows, 1, "mock")
            self.assertEqual(comparison["paired_attempted"], 1)
            self.assertIsNone(comparison["accuracy_delta_after_minus_before"])
            self.assertIsNone(comparison["total_tokens_delta_after_minus_before"])
            self.assertEqual(comparison["model_calls_delta_after_minus_before"], 0)

    def test_real_labelled_stub_computes_paired_token_and_accuracy_deltas(self):
        _, _, before, after = prune_checkpoint(self.checkpoint(), self.config, 0.25, parent_sha256="parent")
        settings = ModelSettings(backend="deepseek", model="offline-test")
        usage = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
        adapter = ModelAdapter(FakeMessagesListChatModel(responses=[AIMessage(content="The answer is 5", usage_metadata=usage)]), settings)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "eval"
            run_baselines([GSM8KSample("gsm8k:test:0", "test", 0, "Q", "5")], self.config, settings, output, {},
                          baselines=("before", "after"), learned_mode="original-fixed", allow_real_api=True,
                          adapter_factory=lambda: adapter, policy_variants={"before": before, "after": after})
            rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
            comparison = compare_pairs(rows, 1, "deepseek")
            self.assertEqual(comparison["accuracy_delta_after_minus_before"], 0)
            self.assertEqual(comparison["total_tokens_delta_after_minus_before"], 0)
            self.assertEqual(comparison["token_saving_fraction"], 0)
            with self.assertRaises(ValueError):
                run_baselines([GSM8KSample("gsm8k:test:0", "test", 0, "Q", "5")], self.config, settings,
                              Path(directory) / "bad", {}, baselines=("before", "after"), learned_mode="sampled",
                              policy_variants={"before": before, "after": after}, allow_real_api=True)

    def test_failed_after_run_is_not_counted_as_efficiency_saving(self):
        _, _, before, after = prune_checkpoint(self.checkpoint(), self.config, 0.25, parent_sha256="parent")
        settings = ModelSettings(backend="deepseek", model="offline-test")
        adapter = ModelAdapter(FakeMessagesListChatModel(responses=[AIMessage(content="The answer is 5")]), settings)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "eval"
            with patch.object(adapter, "invoke", side_effect=[*[AIMessage(content="The answer is 5")] * 6,
                                                             RuntimeError("SECRET_KEY")]):
                report = run_baselines([GSM8KSample("gsm8k:test:0", "test", 0, "Q", "5")], self.config,
                    settings, output, {}, baselines=("before", "after"), learned_mode="original-fixed",
                    allow_real_api=True, adapter_factory=lambda: adapter, policy_variants={"before": before, "after": after})
            self.assertEqual(report["status"], "failed")
            rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
            comparison = compare_pairs(rows, 1, "deepseek")
            self.assertEqual(comparison["paired_attempted"], 1)
            self.assertEqual(comparison["efficiency_pairs"], 0)
            self.assertIsNone(comparison["model_calls_delta_after_minus_before"])
            self.assertIsNone(comparison["total_tokens_delta_after_minus_before"])
            self.assertNotIn("SECRET_KEY", (output / "predictions.jsonl").read_text())

    def test_named_pairs_exclude_unrelated_failures_and_track_missing_groups(self):
        def row(sample, baseline, status, correct=False, tokens=12):
            return {"sample_id": sample, "baseline": baseline, "status": status,
                    "correct": correct, "trace": [{"usage": {"total_tokens": tokens}}],
                    "model_calls": 5, "elapsed_seconds": 1}

        rows = [row("0", "before", "success", True, tokens=20),
                row("0", "random", "success"),
                row("0", "after", "failed"),
                row("1", "before", "success", True)]
        random_vs_full = compare_pairs(rows, 2, "deepseek", after="random")
        self.assertEqual(random_vs_full["paired_successful"], 1)
        self.assertEqual(random_vs_full["missing_pairs"], 1)
        self.assertEqual(random_vs_full["accuracy_delta_after_minus_before"], -1)
        self.assertEqual(random_vs_full["total_tokens_delta_after_minus_before"], -8)
        learned_vs_random = compare_pairs(rows, 2, "deepseek", before="random")
        self.assertEqual(learned_vs_random["paired_attempted"], 1)
        self.assertEqual(learned_vs_random["efficiency_pairs"], 0)
        self.assertIsNone(learned_vs_random["total_tokens_delta_after_minus_before"])

    @unittest.skipUnless((ROOT / "data/gsm8k/test.jsonl").exists(), "官方数据需先准备")
    def test_cli_plan_no_models_and_prune_only_preserves_source_file(self):
        import scripts.prune_topology as entry
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pt"
            torch.save(self.checkpoint(), source)
            before_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            with patch.object(sys, "argv", ["prune.py", "--checkpoint", str(source), "--deepseek", "--plan", "--limit", "2"]), \
                    patch.object(entry, "create_model_adapter") as factory, patch("dotenv.load_dotenv") as load:
                runpy.run_path(str(ROOT / "prune.py"), run_name="__main__")
            factory.assert_not_called()
            load.assert_not_called()
            output = root / "pruned"
            with patch.object(sys, "argv", ["prune.py", "--checkpoint", str(source), "--prune-only", "--output-dir", str(output)]):
                runpy.run_path(str(ROOT / "prune.py"), run_name="__main__")
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before_hash)
            checkpoint = load_training_checkpoint(output / "checkpoint.pt")
            self.assertTrue(checkpoint["evaluation_only"])


if __name__ == "__main__":
    unittest.main()
