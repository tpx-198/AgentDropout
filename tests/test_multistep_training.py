"""验证跨步恢复、失败不更新、测试集隔离和冻结拓扑评测。"""

import copy
import io
import json
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
from ad_l.training import load_training_checkpoint, policy_from_checkpoint, train_multistep

ROOT = Path(__file__).resolve().parents[1]


class MultiStepTests(unittest.TestCase):
    def setUp(self):
        self.network = patch("socket.socket.connect", side_effect=AssertionError("offline test"))
        self.network.start()
        self.quiet = redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.pool = [GSM8KSample(f"gsm8k:train:{i}", "train", i, f"Question {i}: 3 + 2?", "5") for i in range(5)]
        self.settings = ModelSettings()
        self.config = load_fixed_graph_config(ROOT / "configs/fixed_graph_offline.json")
        self.context = {"dataset": {"name": "gsm8k", "split": "train", "sha256": "fixture"},
                        "pool_sample_ids": [sample.sample_id for sample in self.pool],
                        "hyperparameters": {"seed": 2, "batch_size": 2, "lr": 0.1, "initial_probability": 0.4,
                                            "temperature": 1.0, "delta": 0.1, "limit": None},
                        "settings": asdict(self.settings), "config": self.config.to_dict()}

    def tearDown(self):
        self.quiet.__exit__(None, None, None)
        self.network.stop()

    def run_train(self, directory, steps, checkpoint=None, **kwargs):
        return train_multistep(self.pool, self.config, self.settings, self.context,
                              directory, steps, checkpoint=checkpoint, **kwargs)

    def test_resumed_matches_uninterrupted_parameters_optimizer_rng_and_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_train(root / "full", 3)
            self.run_train(root / "first", 2)
            checkpoint = load_training_checkpoint(root / "first/checkpoint.pt")
            self.run_train(root / "resume", 3, checkpoint)
            full = load_training_checkpoint(root / "full/checkpoint.pt")
            resumed = load_training_checkpoint(root / "resume/checkpoint.pt")
            for key in full["model_state_dict"]:
                torch.testing.assert_close(full["model_state_dict"][key], resumed["model_state_dict"][key], rtol=0, atol=0)
            for key in full["optimizer_state_dict"]["state"]:
                for field, value in full["optimizer_state_dict"]["state"][key].items():
                    torch.testing.assert_close(value, resumed["optimizer_state_dict"]["state"][key][field], rtol=0, atol=0)
            self.assertTrue(torch.equal(full["generator_state"], resumed["generator_state"]))
            self.assertEqual(resumed["cursor"], 6)
            def history(name):
                return [json.loads(line) for line in (root / name / "history.jsonl").read_text().splitlines()]
            full_history = history("full")
            combined = history("first") + history("resume")
            for first, second in zip(full_history, combined):
                self.assertEqual(first["sample_ids"], second["sample_ids"])
                self.assertEqual(first["loss"], second["loss"])
                self.assertEqual([e["sampled_rounds"] for e in first["episodes"]], [e["sampled_rounds"] for e in second["episodes"]])

    def test_dataset_settings_and_cursor_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_train(root / "first", 1)
            checkpoint = load_training_checkpoint(root / "first/checkpoint.pt")
            changed = copy.deepcopy(self.context)
            changed["dataset"]["sha256"] = "changed"
            with self.assertRaises(ValueError):
                train_multistep(self.pool, self.config, self.settings, changed, root / "bad", 2, checkpoint=checkpoint)
            self.assertFalse((root / "bad").exists())
            with self.assertRaises(ValueError):
                train_multistep([GSM8KSample("test", "test", 0, "Q", "5")], self.config,
                                self.settings, self.context, root / "test", 1)
            checkpoint["cursor"] = 123
            torch.save(checkpoint, root / "corrupt.pt")
            with self.assertRaises(ValueError):
                load_training_checkpoint(root / "corrupt.pt")

    def test_failure_keeps_last_successful_checkpoint_for_retry(self):
        self.context["hyperparameters"]["batch_size"] = 1
        adapter = ModelAdapter(FakeMessagesListChatModel(responses=[AIMessage(content="The answer is 5")]), self.settings)
        original = adapter.invoke
        counter = [0]
        def invoke(messages):
            counter[0] += 1
            if counter[0] == 7:
                raise RuntimeError("SECRET_TOKEN")
            return original(messages)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(adapter, "invoke", side_effect=invoke):
                report = self.run_train(root / "failed", 2, adapter_factory=lambda: adapter)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["real_api_calls"], 0)
            self.assertEqual(report["model_calls_this_run"], 7)
            checkpoint = load_training_checkpoint(root / "failed/checkpoint.pt")
            self.assertEqual(checkpoint["step"], 1)
            self.assertEqual(checkpoint["cursor"], 1)
            self.assertNotIn("SECRET_TOKEN", (root / "failed/history.jsonl").read_text())
            self.run_train(root / "retry", 2, checkpoint)
            self.run_train(root / "full", 2)
            resumed = load_training_checkpoint(root / "retry/checkpoint.pt")
            full = load_training_checkpoint(root / "full/checkpoint.pt")
            for name in full["model_state_dict"]:
                torch.testing.assert_close(full["model_state_dict"][name], resumed["model_state_dict"][name], rtol=0, atol=0)

    def test_checkpoint_evaluation_is_frozen_and_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_train(root / "train", 1)
            checkpoint = load_training_checkpoint(root / "train/checkpoint.pt")
            policy = policy_from_checkpoint(checkpoint).eval().requires_grad_(False)
            before = {name: value.clone() for name, value in policy.state_dict().items()}
            samples = [GSM8KSample("gsm8k:test:0", "test", 0, "3 + 2?", "5")]
            for mode in ("sampled", "original-fixed"):
                report = run_baselines(samples, self.config, self.settings, root / mode, {},
                    baselines=("single", "fixed", "learned"), learned_policy=policy,
                    learned_generator=torch.Generator().manual_seed(9), learned_mode=mode)
                self.assertEqual(report["summary"]["learned"]["model_calls"], 5)
                rows = [json.loads(line) for line in (root / mode / "predictions.jsonl").read_text().splitlines()]
                self.assertEqual([row["sample_id"] for row in rows], [samples[0].sample_id] * 3)
                self.assertEqual(rows[-1]["learned_mode"], mode)
            for name, value in policy.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            self.assertTrue(all(parameter.grad is None for parameter in policy.parameters()))


if __name__ == "__main__":
    unittest.main()
