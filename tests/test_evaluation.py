"""批量基线：可复核抽样、原判分兼容、无答案泄漏、失败持久化和用量统计。"""

import io
import json
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from ad_l.datasets import GSM8KSample, load_gsm8k, sample_gsm8k
from ad_l.evaluation import SCORER, run_baselines, score_answer
from ad_l.fixed_graph import load_fixed_graph_config
from ad_l.models import ModelAdapter, ModelSettings

ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.network = patch("socket.socket.connect", side_effect=AssertionError("offline test"))
        self.network.start()
        self.config = load_fixed_graph_config(ROOT / "configs/fixed_graph_offline.json")

    def tearDown(self):
        self.network.stop()

    def samples(self):
        return [GSM8KSample(f"gsm8k:test:{i}", "test", i, f"QUESTION_{i}", "5") for i in range(3)]

    def test_loader_and_sampling_are_stable_and_do_not_change_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            path.write_text("\n".join(json.dumps({"question": f"Q{i}", "answer": "SECRET_SOLUTION\n#### 1,234"})
                                      for i in range(30)) + "\n")
            samples, info = load_gsm8k(path, "test")
            self.assertEqual(samples[0].target, "1234")
            self.assertFalse(info["verified_official"])
            self.assertEqual(sample_gsm8k(samples, 7, 42), sample_gsm8k(samples, 7, 42))
            self.assertNotEqual(sample_gsm8k(samples, 7, 42), sample_gsm8k(samples, 7, 43))
            self.assertTrue(all(s.split == "test" for s in sample_gsm8k(samples, 7, 42)))
            with self.assertRaises(ValueError):
                load_gsm8k(path, "test", verify_official=True)
            for limit in [0, 31, -1]:
                with self.assertRaises(ValueError):
                    sample_gsm8k(samples, limit, 0)

    def test_malformed_labels_and_empty_dataset_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            for text in ["", "{}", '{"question":"Q", "answer":"5"}',
                         '{"question":"Q", "answer":"work\\n#### nan"}']:
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_gsm8k(path, "test")

    def test_scorer_preserves_original_rule_and_marks_errors(self):
        self.assertEqual(SCORER, "agentdropout_gsm_get_predict_float_v1")
        self.assertTrue(score_answer("The answer is 5.", "5")["correct"])
        self.assertFalse(score_answer("The answer is -12", "-12")["correct"])
        self.assertFalse(score_answer("The answer is 1.5", "1.5")["correct"])
        self.assertTrue(score_answer("No numbers", "0")["correct"])  # 原实现的已知局限
        self.assertFalse(score_answer("The answer is 5", "5", "failed")["correct"])
        self.assertIsNotNone(score_answer("boxed", "5")["scoring_error"])

    def test_real_labelled_fake_models_measure_usage_and_use_same_samples(self):
        settings = ModelSettings(backend="deepseek", model="offline-test")
        usage = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
        adapter = ModelAdapter(FakeMessagesListChatModel(responses=[AIMessage(content="The answer is 5", usage_metadata=usage)]), settings)
        samples = self.samples()
        samples[0] = GSM8KSample("gsm8k:test:0", "test", 0, "QUESTION_0", "999999987123")
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "run"
            report = run_baselines(samples, self.config, settings, output, {"scorer": SCORER},
                                   allow_real_api=True, adapter_factory=lambda: adapter)
            rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            for baseline, count in [("single", 3), ("fixed", 15)]:
                summary = report["summary"][baseline]
                self.assertEqual(summary["model_calls"], count)
                self.assertEqual(summary["real_api_calls"], count)
                self.assertEqual(summary["usage"]["total_tokens"], count * 12)
                self.assertEqual(summary["accuracy"], 2 / 3)
                self.assertEqual([r["sample_id"] for r in rows if r["baseline"] == baseline], [s.sample_id for s in samples])
            for row in rows:
                for call in row["trace"]:
                    messages = json.dumps(call["messages"])
                    self.assertNotIn("999999987123", messages)
                    self.assertNotIn("SECRET_SOLUTION", messages)

    def test_failure_is_saved_and_stops_further_requests(self):
        settings = ModelSettings(backend="deepseek", model="offline-test")
        adapter = ModelAdapter(FakeMessagesListChatModel(responses=[AIMessage(content="The answer is 5")]), settings)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "run"
            with patch.object(adapter, "invoke", side_effect=[AIMessage(content="The answer is 5"),
                              AIMessage(content="The answer is 5"), RuntimeError("SECRET_CREDENTIAL")]) as invoke:
                report = run_baselines(self.samples(), self.config, settings, output, {},
                                       allow_real_api=True, adapter_factory=lambda: adapter)
            self.assertEqual(invoke.call_count, 3)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["summary"]["fixed"]["failures"], 1)
            self.assertEqual(report["summary"]["fixed"]["real_api_calls"], 2)
            self.assertEqual(report["summary"]["fixed"]["unattempted"], 2)
            self.assertEqual(len((output / "predictions.jsonl").read_text().splitlines()), 2)
            self.assertNotIn("SECRET_CREDENTIAL", (output / "predictions.jsonl").read_text())
            self.assertEqual(json.loads((output / "summary.json").read_text())["status"], "failed")

    def test_mock_metrics_are_not_reported_as_model_accuracy(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "run"
            report = run_baselines(self.samples(), self.config, ModelSettings(), output, {})
            for summary in report["summary"].values():
                self.assertIsNone(summary["accuracy"])
                self.assertIsNone(summary["usage"])
                self.assertEqual(summary["real_api_calls"], 0)
                self.assertEqual(summary["mock_diagnostic_correct"], 3)
            with self.assertRaises(FileExistsError):
                run_baselines(self.samples(), self.config, ModelSettings(), output, {})

    def test_real_runs_need_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_baselines(self.samples(), self.config, ModelSettings(backend="deepseek", model="test"),
                              Path(directory) / "run", {})
            self.assertFalse((Path(directory) / "run").exists())

    def test_factory_settings_mismatch_is_saved_before_any_call(self):
        wrong = ModelAdapter(FakeMessagesListChatModel(responses=[AIMessage(content="The answer is 5")]),
                             ModelSettings(model="wrong-settings"))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "run"
            with patch.object(wrong, "invoke") as invoke:
                report = run_baselines(self.samples(), self.config, ModelSettings(), output, {},
                                       adapter_factory=lambda: wrong)
            invoke.assert_not_called()
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["summary"]["single"]["model_calls"], 0)
            self.assertEqual(json.loads((output / "predictions.jsonl").read_text())["setup_error"]["type"], "ValueError")

    @unittest.skipUnless((ROOT / "data/gsm8k/test.jsonl").exists(), "官方数据需先准备")
    def test_official_data_and_cli_plan_do_not_create_models(self):
        import scripts.run_evaluation as entry

        for split in ["train", "test"]:
            samples, info = load_gsm8k(ROOT / f"data/gsm8k/{split}.jsonl", split, verify_official=True)
            self.assertEqual(len(samples), 7473 if split == "train" else 1319)
            self.assertTrue(info["verified_official"])
        stdout = io.StringIO()
        with patch.object(sys, "argv", ["evaluate.py", "--deepseek", "--limit", "2", "--plan"]), \
                patch.object(entry, "create_model_adapter") as factory, \
                patch("dotenv.load_dotenv") as load, redirect_stdout(stdout):
            runpy.run_path(str(ROOT / "evaluate.py"), run_name="__main__")
        factory.assert_not_called()
        load.assert_not_called()
        self.assertEqual(json.loads(stdout.getvalue())["planned_real_api_calls"], 12)


if __name__ == "__main__":
    unittest.main()
