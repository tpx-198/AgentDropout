import contextlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from ad_l.models import create_model
from ad_l.roles import create_team
from ad_l.topology import EdgePolicy, TopologyConfig, load_policy
from ad_l.training import Example, answer_number, load_examples, train_step

ROOT = Path(__file__).resolve().parents[1]


class TrainingTests(unittest.TestCase):
    def setup_training(self):
        policy = EdgePolicy(TopologyConfig(topology='chain'))
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.1)
        team = create_team()
        adapters = {agent.node_id: create_model() for agent in team}
        return policy, optimizer, adapters

    def test_reward_drives_policy_gradient_without_regularization(self):
        policy, optimizer, adapters = self.setup_training()
        before = policy.summary_logits.detach().clone()
        report = train_step(policy, optimizer, [Example('synthetic:0', '3 + 2?', '5')],
                            adapters, torch.Generator().manual_seed(0), sparsity=0)
        self.assertTrue(report['updated'])
        self.assertEqual(report['episodes'][0]['reward'], 1)
        self.assertFalse(torch.equal(before, policy.summary_logits))
        json.dumps(report)
        state = report['episodes'][0]['result']
        self.assertNotIn('log_prob', str(state))
        for parameter in policy.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_zero_reward_without_regularization_does_not_update(self):
        policy, optimizer, adapters = self.setup_training()
        report = train_step(policy, optimizer, [Example('synthetic:0', '3 + 2?', '9')],
                            adapters, torch.Generator().manual_seed(0), sparsity=0)
        self.assertFalse(report['updated'])
        self.assertEqual(report['episodes'][0]['reward'], 0)

    def test_failed_batch_leaves_parameters_optimizer_and_rng_unchanged(self):
        policy, optimizer, adapters = self.setup_training()
        generator = torch.Generator().manual_seed(4)
        rng_before = generator.get_state()
        weights_before = policy.report()
        with patch.object(adapters['agent_1'], 'invoke', side_effect=RuntimeError('failure')):
            result = train_step(policy, optimizer, [Example('synthetic:0', '3 + 2?', '5')], adapters, generator)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(weights_before, policy.report())
        torch.testing.assert_close(rng_before, generator.get_state())
        self.assertEqual(optimizer.state_dict()['state'], {})

    def test_numeric_scoring_preserves_signs_and_decimals(self):
        self.assertEqual(str(answer_number('The answer is -1,234.50')), '-1234.50')
        self.assertEqual(str(answer_number(r'\boxed{0.5}')), '0.5')
        self.assertIsNone(answer_number('Some unrelated number 42'))
        self.assertIsNone(answer_number('The answer is unknown'))

    def test_loader_uses_train_split_and_stable_selection(self):
        examples, dataset = load_examples(ROOT / 'data/gsm8k/train.jsonl', 3, 0)
        repeated, _ = load_examples(ROOT / 'data/gsm8k/train.jsonl', 3, 0)
        self.assertEqual(examples, repeated)
        self.assertEqual(dataset['records'], 7473)
        self.assertTrue(all(example.sample_id.startswith('gsm8k:train:') for example in examples))

    def test_plan_does_not_create_models_or_load_credentials(self):
        for filename in ('run.py', 'train.py'):
            spec = importlib.util.spec_from_file_location('command_entry', ROOT / filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            output = io.StringIO()
            with patch.object(sys, 'argv', [filename, '--plan', '--deepseek']), \
                    patch.object(module, 'adapters_from_args', side_effect=AssertionError('client created')), \
                    patch('dotenv.load_dotenv', side_effect=AssertionError('dotenv loaded')), contextlib.redirect_stdout(output):
                module.main()
            plan = json.loads(output.getvalue())
            self.assertEqual(plan['planned_real_api_calls'], 9 if filename == 'run.py' else 54)

    def test_training_checkpoint_and_run_work_without_tests_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'policy.pt'
            spec = importlib.util.spec_from_file_location('train_entry', ROOT / 'train.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with patch.object(sys, 'argv', ['train.py', '--steps', '1', '--limit', '2', '--output', str(checkpoint)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                module.main()
            policy, metadata = load_policy(checkpoint)
            self.assertEqual(metadata['real_api_calls'], 0)
            self.assertEqual(metadata['metrics_kind'], 'mock_diagnostic')
            self.assertEqual(policy.config.agents, 5)
            standalone = Path(directory) / 'standalone'
            standalone.mkdir()
            shutil.copytree(ROOT / 'ad_l', standalone / 'ad_l')
            shutil.copyfile(ROOT / 'run.py', standalone / 'run.py')
            output = io.StringIO()
            standalone_path = [str(standalone), *[entry for entry in sys.path if Path(entry or '.').resolve() != ROOT]]
            with patch.dict(sys.modules), patch.object(sys, 'path', standalone_path), \
                    patch.object(sys, 'argv', ['run.py', '--checkpoint', str(checkpoint), '--async']), \
                    contextlib.redirect_stdout(output):
                for name in list(sys.modules):
                    if name == 'ad_l' or name.startswith('ad_l.'):
                        del sys.modules[name]
                spec = importlib.util.spec_from_file_location('standalone_entry', standalone / 'run.py')
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.main()
                self.assertEqual(Path(sys.modules['ad_l'].__file__).parent, standalone / 'ad_l')
            self.assertIn('real calls 0', output.getvalue())


if __name__ == '__main__':
    unittest.main()
