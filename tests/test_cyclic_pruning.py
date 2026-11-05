"""周期剪枝：更新后执行、失败不触发、恢复不重复或漏剪。"""

import copy
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
from ad_l.fixed_graph import load_fixed_graph_config
from ad_l.models import ModelAdapter, ModelSettings
from ad_l.training import load_training_checkpoint, train_multistep, pruning_schedule

ROOT = Path(__file__).resolve().parents[1]


class CyclicPruningTests(unittest.TestCase):
    def setUp(self):
        self.network = patch("socket.socket.connect", side_effect=AssertionError("offline test"))
        self.network.start()
        self.quiet = redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.settings = ModelSettings()
        self.config = load_fixed_graph_config(ROOT / "configs/fixed_graph_offline.json")
        self.pool = [GSM8KSample(f"gsm8k:train:{i}", "train", i, f"Question {i}: 3 + 2?", "5") for i in range(5)]
        self.context = {"dataset": {"split": "train", "sha256": "fixture"},
            "pool_sample_ids": [sample.sample_id for sample in self.pool],
            "settings": asdict(self.settings), "config": self.config.to_dict(),
            "hyperparameters": {"seed": 2, "batch_size": 1, "lr": 0.1, "initial_probability": 0.4,
                                "temperature": 1.0, "delta": 0.1, "prune_every": 2, "prune_rate": 0.25,
                                "prune_until": 4}}

    def tearDown(self):
        self.quiet.__exit__(None, None, None)
        self.network.stop()

    def train(self, output, steps, checkpoint=None, **kwargs):
        return train_multistep(self.pool, self.config, self.settings, self.context, output, steps,
                              checkpoint=checkpoint, **kwargs)

    def assert_same_checkpoint(self, first, second):
        for name, value in first["model_state_dict"].items():
            torch.testing.assert_close(value, second["model_state_dict"][name], rtol=0, atol=0)
        for index, state in first["optimizer_state_dict"]["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, second["optimizer_state_dict"]["state"][index][name], rtol=0, atol=0)
        self.assertTrue(torch.equal(first["generator_state"], second["generator_state"]))
        self.assertEqual(first["cursor"], second["cursor"])
        self.assertEqual(first["pruning_state"], second["pruning_state"])

    def test_continuous_and_boundary_resume_match_including_optimizer_and_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = self.train(root / "full", 6)
            self.train(root / "first", 2)
            boundary = load_training_checkpoint(root / "first/checkpoint.pt")
            self.assertEqual([event["step"] for event in boundary["pruning_state"]["events"]], [2])
            self.train(root / "resumed", 6, boundary)
            full = load_training_checkpoint(root / "full/checkpoint.pt")
            resumed = load_training_checkpoint(root / "resumed/checkpoint.pt")
            self.assert_same_checkpoint(full, resumed)
            self.assertEqual([event["step"] for event in full["pruning_state"]["events"]], [2, 4])
            self.assertEqual(summary["pruning_events_total"], 2)
            history = [json.loads(line) for line in (root / "full/history.jsonl").read_text().splitlines()]
            self.assertIsNone(history[0]["pruning_event"])
            self.assertEqual(history[1]["pruning_event"]["phase"], "after_optimizer_step")
            self.assertEqual(history[1]["pruning_event"]["changes"]["spatial"]["actual_removed"], 1)
            for report in history[2:]:
                for episode in report["episodes"]:
                    self.assertTrue(all(not item["spatial_edges"] for item in episode["sampled_rounds"]))
            self.assertEqual(history[3]["pruning_event"]["changes"]["spatial"]["actual_removed"], 0)

    def test_failure_on_due_step_does_not_prune_and_retry_matches_full_run(self):
        adapter = ModelAdapter(FakeMessagesListChatModel(responses=[AIMessage(content="The answer is 5")]), self.settings)
        original = adapter.invoke
        count = [0]
        def invoke(messages):
            count[0] += 1
            if count[0] == 7:
                raise RuntimeError("SECRET_TOKEN")
            return original(messages)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(adapter, "invoke", side_effect=invoke):
                summary = self.train(root / "failed", 3, adapter_factory=lambda: adapter)
            checkpoint = load_training_checkpoint(root / "failed/checkpoint.pt")
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(checkpoint["step"], 1)
            self.assertEqual(checkpoint["pruning_state"]["events"], [])
            self.train(root / "retry", 3, checkpoint)
            self.train(root / "full", 3)
            self.assert_same_checkpoint(load_training_checkpoint(root / "retry/checkpoint.pt"),
                                        load_training_checkpoint(root / "full/checkpoint.pt"))

    def test_crash_before_boundary_checkpoint_replays_update_and_prune_once(self):
        import ad_l.training as training
        save = training.save_training_checkpoint
        def interrupted(path, policy, optimizer, generator, step, cursor, context, **kwargs):
            if step == 2:
                raise OSError("simulated write interruption")
            return save(path, policy, optimizer, generator, step, cursor, context, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(training, "save_training_checkpoint", side_effect=interrupted):
                with self.assertRaises(OSError):
                    self.train(root / "crashed", 3)
            checkpoint = load_training_checkpoint(root / "crashed/checkpoint.pt")
            self.assertEqual(checkpoint["step"], 1)
            self.train(root / "retry", 3, checkpoint)
            self.train(root / "full", 3)
            self.assert_same_checkpoint(load_training_checkpoint(root / "retry/checkpoint.pt"),
                                        load_training_checkpoint(root / "full/checkpoint.pt"))

    def test_missing_event_or_changed_schedule_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.train(root / "first", 2)
            checkpoint = load_training_checkpoint(root / "first/checkpoint.pt")
            corrupt = copy.deepcopy(checkpoint)
            corrupt["pruning_state"]["events"] = []
            torch.save(corrupt, root / "corrupt.pt")
            with self.assertRaisesRegex(ValueError, "剪枝事件"):
                load_training_checkpoint(root / "corrupt.pt")
            self.context["hyperparameters"]["prune_every"] = 1
            with self.assertRaises(ValueError):
                self.train(root / "bad", 3, checkpoint)
            self.assertFalse((root / "bad").exists())
        for update in [{"prune_every": -1}, {"prune_rate": 1.2}, {"prune_until": 0}]:
            with self.assertRaises(ValueError):
                pruning_schedule({"hyperparameters": {**self.context["hyperparameters"], **update}})

    @unittest.skipUnless((ROOT / "data/gsm8k/train.jsonl").exists(), "官方数据需先准备")
    def test_cli_plan_has_absolute_pruning_steps_and_no_model_calls(self):
        import scripts.train_topology as entry
        stdout = io.StringIO()
        argv = ["train.py", "--steps", "6", "--prune-every", "2", "--prune-until", "4", "--deepseek", "--plan"]
        with patch.object(sys, "argv", argv), patch.object(entry, "create_model_adapter") as factory, \
                patch("dotenv.load_dotenv") as load, redirect_stdout(stdout):
            runpy.run_path(str(ROOT / "train.py"), run_name="__main__")
        factory.assert_not_called()
        load.assert_not_called()
        self.assertEqual(json.loads(stdout.getvalue())["pruning_steps_this_run"], [2, 4])


if __name__ == "__main__":
    unittest.main()
