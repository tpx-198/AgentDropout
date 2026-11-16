import argparse
import asyncio
import contextlib
from dataclasses import asdict
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ad_l.cli import adapters_from_args, add_arguments
from ad_l.model_config import create_agent_models, load_agent_config, model_call_plan
from ad_l.roles import create_team
from ad_l.system import build_system
from ad_l.topology import EdgePolicy, TopologyConfig, load_policy

ROOT = Path(__file__).resolve().parents[1]


class AgentAPIConfigTests(unittest.TestCase):
    def mock_config(self, team):
        return {agent.node_id: {"backend": "mock", "mock_response": f"{agent.node_id} The answer is 5"} for agent in team}

    def load_config(self, directory, values, team):
        path = Path(directory) / 'agents.json'
        path.write_text(json.dumps(values))
        return load_agent_config(path, team)

    def test_different_clients_models_keys_and_endpoints_in_sync_and_async(self):
        team = create_team()
        values = self.mock_config(team)
        values['agent_1'] = {'backend': 'deepseek', 'model_env': 'FIRST_MODEL', 'api_key_env': 'FIRST_KEY',
                             'base_url': 'https://first.example/v1', 'temperature': 0.3, 'max_tokens': 500, 'timeout': 12.0}
        values['final'] = {'backend': 'openai', 'model': 'second-model', 'api_key_env': 'SECOND_KEY',
                           'base_url': 'https://second.example/v1', 'timeout': 15.0}
        environment = {'FIRST_MODEL': 'first-model', 'FIRST_KEY': 'private-first-key', 'SECOND_KEY': 'private-second-key'}
        with tempfile.TemporaryDirectory() as directory:
            config = self.load_config(directory, values, team)
            with patch.dict(os.environ, environment, clear=True), \
                    patch('langchain_deepseek.ChatDeepSeek', return_value=FakeListChatModel(responses=['The answer is 5'])) as first, \
                    patch('langchain_openai.ChatOpenAI', return_value=FakeListChatModel(responses=['The answer is 5'])) as second:
                before = dict(os.environ)
                adapters, settings = create_agent_models(config, allow_real_api=True)
                self.assertEqual(dict(os.environ), before)
                self.assertEqual(first.call_args.kwargs['api_key'], 'private-first-key')
                self.assertEqual(second.call_args.kwargs['api_key'], 'private-second-key')
                self.assertEqual(first.call_args.kwargs['model'], 'first-model')
                self.assertEqual(second.call_args.kwargs['model'], 'second-model')
                self.assertEqual(first.call_args.kwargs['base_url'], 'https://first.example/v1')
                self.assertEqual(second.call_args.kwargs['base_url'], 'https://second.example/v1')
                self.assertEqual(first.call_args.kwargs['max_retries'], 0)
                self.assertEqual(first.call_args.kwargs['max_tokens'], 500)
                self.assertEqual(first.call_args.kwargs['temperature'], 0.3)
                self.assertEqual(len({id(adapter) for adapter in adapters.values()}), 5)
                execution = EdgePolicy(TopologyConfig()).sample(torch.Generator(), mode='fixed').execution
                graph = build_system(team, execution, adapters, allow_real_api=True)
                for state in (graph.invoke({'task': '3 + 2?'}), asyncio.run(graph.ainvoke({'task': '3 + 2?'}))):
                    self.assertEqual(state['status'], 'success')
                    self.assertEqual(state['real_api_calls'], 3)
                    self.assertNotIn('private-first-key', str(state))
                    self.assertNotIn('private-second-key', str(state))
                serialized = json.dumps({'settings': {node: asdict(value) for node, value in settings.items()}, 'config': config})
                self.assertNotIn('private-first-key', serialized)
                self.assertEqual(model_call_plan(config, team, rounds=2, examples=4)['planned_real_api_calls'], 12)

    def test_real_nodes_need_opt_in_and_all_missing_credentials_fail_before_clients(self):
        team = create_team()
        values = self.mock_config(team)
        values['agent_1'] = {'backend': 'deepseek', 'model': 'model-one', 'api_key_env': 'FIRST_KEY'}
        values['final'] = {'backend': 'deepseek', 'model': 'model-two', 'api_key_env': 'LAST_KEY'}
        with tempfile.TemporaryDirectory() as directory:
            config = self.load_config(directory, values, team)
            with patch('dotenv.load_dotenv', side_effect=AssertionError('dotenv read before opt-in')):
                with self.assertRaisesRegex(ValueError, 'allow-real-api'):
                    create_agent_models(config, env_file=ROOT / '.env')
            with patch.dict(os.environ, {'FIRST_KEY': 'private-first-key'}, clear=True), \
                    patch('langchain_deepseek.ChatDeepSeek') as client:
                with self.assertRaisesRegex(ValueError, 'LAST_KEY'):
                    create_agent_models(config, allow_real_api=True)
                client.assert_not_called()

    def test_each_config_must_cover_current_team_and_reject_ambiguous_or_secret_fields(self):
        team = create_team()
        bad_specs = [
            {'backend': 'deepseek', 'model': 'x', 'api_key': 'secret'},
            {'backend': 'deepseek', 'model': 'x', 'model_env': 'MODEL'},
            {'backend': 'deepseek'},
            {'backend': 'deepseek', 'model': 'x', 'api_key_env': 'bad name'},
            {'backend': 'deepseek', 'model': 'x', 'base_url': 'https://name:secret@example.com/v1'},
            {'backend': 'openai', 'model': 'x', 'base_url': 'https://example.com/v1?api_key=secret'},
            {'backend': 'unknown', 'model': 'x'},
        ]
        with tempfile.TemporaryDirectory() as directory:
            for spec in bad_specs:
                values = self.mock_config(team)
                values['agent_1'] = spec
                with self.subTest(spec=spec), self.assertRaises(ValueError):
                    self.load_config(directory, values, team)
            values = self.mock_config(team)
            del values['final']
            with self.assertRaisesRegex(ValueError, 'node IDs'):
                self.load_config(directory, values, team)
            self.load_config(directory, self.mock_config(create_team(8)), create_team(8))
            path = Path(directory) / 'agents.json'
            path.write_text('{"agent_1": {}, "agent_1": {}}')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                load_agent_config(path, team)

    def test_plan_supports_template_without_reading_env_or_creating_clients(self):
        for filename in ('run.py', 'train.py'):
            spec = importlib.util.spec_from_file_location('api_plan_entry', ROOT / filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            output = io.StringIO()
            args = [filename, '--api-config', str(ROOT / 'configs/agents.json'), '--plan']
            with patch.object(sys, 'argv', args), patch.dict(os.environ, {}, clear=True), \
                    patch('dotenv.load_dotenv', side_effect=AssertionError('dotenv loaded')), \
                    patch.object(module, 'adapters_from_args', side_effect=AssertionError('client created')), \
                    contextlib.redirect_stdout(output):
                module.main()
            plan = json.loads(output.getvalue())
            self.assertEqual(plan['planned_real_api_calls'], 9 if filename == 'run.py' else 54)
            self.assertEqual(set(plan['agent_api_config']['nodes']), {'agent_1', 'agent_2', 'agent_3', 'agent_4', 'final'})
            self.assertEqual(plan['planned_model_calls_by_agent']['final'], 1 if filename == 'run.py' else 6)

    def test_mock_config_drives_cli_outputs_and_training_checkpoint(self):
        team = create_team()
        with tempfile.TemporaryDirectory() as directory:
            config = self.load_config(directory, self.mock_config(team), team)
            for filename in ('run.py', 'train.py'):
                output_path = Path(directory) / ('result.json' if filename == 'run.py' else 'policy.pt')
                options = ['--async'] if filename == 'run.py' else ['--steps', '1', '--limit', '1']
                args = [filename, '--api-config', config['path'], '--output', str(output_path), *options]
                spec = importlib.util.spec_from_file_location('api_config_entry', ROOT / filename)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                with patch.object(sys, 'argv', args), \
                        patch('dotenv.load_dotenv', side_effect=AssertionError('mock loaded dotenv')), \
                        contextlib.redirect_stdout(io.StringIO()):
                    module.main()
                if filename == 'run.py':
                    record = json.loads(output_path.read_text())
                    self.assertTrue(record['result']['answer'].startswith('final'))
                else:
                    _, record = load_policy(output_path)
                    self.assertEqual(record['metrics_kind'], 'mock_diagnostic')
                self.assertEqual(set(record['settings']), {agent.node_id for agent in team})
                self.assertEqual(record['agent_api_config']['sha256'], config['sha256'])

    def test_environment_overrides_dotenv_without_cross_agent_key_replacement(self):
        team = create_team()
        values = self.mock_config(team)
        values['agent_1'] = {'backend': 'deepseek', 'model_env': 'FIRST_MODEL', 'api_key_env': 'FIRST_KEY'}
        with tempfile.TemporaryDirectory() as directory:
            config = self.load_config(directory, values, team)
            env_file = Path(directory) / '.env'
            env_file.write_text('FIRST_MODEL=dotenv-model\nFIRST_KEY=dotenv-key\n')
            with patch.dict(os.environ, {'FIRST_MODEL': 'environment-model', 'FIRST_KEY': 'environment-key'}, clear=True), \
                    patch('langchain_deepseek.ChatDeepSeek', return_value=FakeListChatModel(responses=['The answer is 5'])) as client:
                create_agent_models(config, allow_real_api=True, env_file=env_file)
                self.assertEqual(client.call_args.kwargs['model'], 'environment-model')
                self.assertEqual(client.call_args.kwargs['api_key'], 'environment-key')


if __name__ == '__main__':
    unittest.main()
