import copy
from email.message import Message
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from ad_l.models import ModelAdapter
from ad_l.platform_service import PlatformService, SettingsStore, default_config
from ad_l.roles import PROMPTS
from ad_l.topology import load_policy
from platform_server import make_handler


class PlatformTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for dataset in ('gsm8k', 'aqua', 'svamp', 'multiarith'):
            directory = self.root / 'data' / dataset
            directory.mkdir(parents=True)
            for split in ('train', 'test'):
                rows = []
                for index in range(4):
                    row = {'id': f'{dataset}:{split}:{index}', 'dataset': dataset, 'split': split,
                           'question': f'{dataset} {split} question {index}: 3 + 2?', 'target': '5', 'answer_type': 'numeric'}
                    if dataset == 'aqua':
                        row.update(answer_type='choice', target='A', choices=['five', 'six'])
                    rows.append(row)
                (directory / f'{split}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        self.service = PlatformService(self.root)
        self.config = default_config()
        self.config['training'].update(limit=2, steps=2, batch_size=2)
        self.config['evaluation'].update(limit=2)

    def tearDown(self):
        if self.service.active:
            self.service.cancel(self.service.active)
            self.wait_for_idle()
        self.temporary.cleanup()

    def wait_for_idle(self):
        deadline = time.monotonic() + 20
        while self.service.active is not None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIsNone(self.service.active, 'background job did not finish')

    def test_settings_persist_keys_privately_and_support_clear(self):
        public = self.service.store.save(self.config, {'agent_1': 'private-test-credential'})
        self.assertTrue(public['key_present']['agent_1'])
        self.assertNotIn('private-test-credential', json.dumps(public))
        self.assertEqual(os.stat(self.service.store.path).st_mode & 0o777, 0o600)
        reopened = SettingsStore(self.service.store.path)
        self.assertTrue(reopened.public()['key_present']['agent_1'])
        public = reopened.save(self.config, clear_keys=['agent_1'])
        self.assertFalse(public['key_present']['agent_1'])
        with self.assertRaises(ValueError):
            reopened.save(self.config, {'final': 'bad\nvalue'})

    def test_training_then_paired_evaluation_uses_heldout_samples_and_freezes_weights(self):
        self.service.store.save(self.config)
        job = self.service.submit('train', self.config)
        self.wait_for_idle()
        result = self.service.public_job(job['id'])
        self.assertEqual(result['status'], 'completed', result['message'])
        self.assertEqual(result['completed_steps'], 2)
        self.assertEqual(result['summary']['model_calls'], 36)
        self.assertIsNone(result['summary']['accuracy'])
        self.assertEqual(result['summary']['real_api_calls'], 0)
        self.assertTrue(all(row['group'] == 'training' for row in result['rows']))
        identity = f"job:{job['id']}"
        config = copy.deepcopy(self.config)
        config['evaluation']['checkpoint'] = identity
        checkpoint = self.service._checkpoint(identity)
        before = checkpoint.read_bytes()
        job = self.service.submit('evaluate', config)
        self.wait_for_idle()
        result = self.service.public_job(job['id'])
        self.assertEqual(result['status'], 'completed', result['message'])
        self.assertEqual(result['progress'], 16)
        self.assertEqual(checkpoint.read_bytes(), before)
        rows = self.service.export(job['id'])['details']['rows']
        baseline = [row['sample_id'] for row in rows if row['group'] == 'baseline']
        trained = [row['sample_id'] for row in rows if row['group'] == 'trained']
        self.assertEqual(baseline, trained)
        self.assertTrue(all(row['split'] == 'test' for row in rows))
        self.assertEqual(result['summary']['trained']['attempted'], 8)

    def test_evaluation_blocks_prompt_overlap_outside_sampled_questions(self):
        self.config['evaluation'].update(datasets=['aqua'], limit=1)
        plan = self.service.plan(self.config, 'evaluate')
        self.assertNotIn('aqua:test:0', plan['sample_ids'])

        decision = {**PROMPTS['decision'], 'few_shot': 'Q: AQUA  TEST\nquestion 0: 3 + 2?\nA: A'}
        with patch.dict(PROMPTS, {'decision': decision}):
            with self.assertRaisesRegex(ValueError, r'FinalRefer \(few_shot\): aqua:test:0'):
                self.service.submit('evaluate', self.config)
        self.assertEqual(self.service.history(), [])

    def test_node_pruning_produces_a_checkpoint_and_reduces_paired_call_budget(self):
        self.config["training"].update(datasets=["gsm8k"], steps=1, batch_size=1,
                                       node_prune_count=1, prune_rate=0)
        self.config["evaluation"].update(datasets=["gsm8k"], limit=1)
        training = self.service.submit("train", self.config)
        self.wait_for_idle()
        self.assertEqual(self.service.public_job(training["id"])["summary"]["model_calls"], 9)
        self.config["evaluation"]["checkpoint"] = f"job:{training['id']}"
        preview = self.service.preview(self.config)
        self.assertEqual([len(round["active"]) for round in preview["execution"]["rounds"]], [3, 3])
        plan = self.service.plan(self.config, "evaluate")
        self.assertEqual(plan["calls_by_group"], {"baseline": 9, "trained": 7})
        job = self.service.submit("evaluate", self.config)
        self.wait_for_idle()
        result = self.service.public_job(job["id"])
        self.assertEqual(result["status"], "completed")
        for group, calls in plan["calls_by_group"].items():
            self.assertEqual(result["summary"][group]["model_calls"], calls)

    def test_evaluation_prunes_an_inactive_real_backend_before_credentials_are_required(self):
        self.config["evaluation"].update(datasets=["gsm8k"], limit=1, node_prune_count=1)
        self.config["models"]["agent_1"].update(backend="openai", model="requires-credentials")
        plan = self.service.plan(self.config, "evaluate")
        self.assertEqual(plan["calls_by_agent"]["agent_1"], 0)
        self.assertEqual(plan["real_api_calls"], 0)
        job = self.service.submit("evaluate", self.config)
        self.wait_for_idle()
        result = self.service.public_job(job["id"])
        self.assertEqual(result["status"], "completed", result["message"])
        self.assertEqual(result["summary"]["current"]["model_calls"], 7)

    def test_aptbench_runs_as_evaluation_only_with_task_specific_prompts(self):
        dataset = 'aptbench_dr'
        directory = self.root / 'data' / dataset
        directory.mkdir()
        rows = [
            {'id': 'aptbench_dr:choice', 'dataset': dataset, 'split': 'test',
             'question': 'Select supported statements.', 'answer_type': 'multi_choice',
             'choices': ['one', 'two', 'three', 'four'], 'target': 'B,D'},
            {'id': 'aptbench_dr:text', 'dataset': dataset, 'split': 'test',
             'question': 'Give the concise research result.', 'answer_type': 'text', 'target': 'PRIVATE_TEXT_RESULT'},
        ]
        (directory / 'test.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        self.assertIn({'name': dataset, 'splits': {'test': 2}}, self.service.dataset_catalog())
        self.config['evaluation'].update(datasets=[dataset], limit=2)
        self.assertEqual(self.service.plan(self.config, 'evaluate')['model_calls'], 18)

        def respond(adapter, messages):
            system = messages[0].content
            if not system.startswith('Select or produce'):
                return AIMessage(content='I reviewed the evidence.')
            if 'comma-separated' in system:
                return AIMessage(content='The answer is D, B')
            return AIMessage(content='The answer is PRIVATE_TEXT_RESULT')

        with patch.object(ModelAdapter, 'invoke', respond):
            job = self.service.submit('evaluate', self.config)
            self.wait_for_idle()
        result = self.service.export(job['id'])
        self.assertEqual(result['summary']['status'], 'completed')
        for row in result['details']['rows']:
            self.assertTrue(row['correct'])
            for call in row['result']['trace']:
                self.assertNotIn('PRIVATE_TEXT_RESULT', str(call['messages']))
                self.assertIn('supplied context', call['messages'][0]['content'])
                original = PROMPTS['decision'] if call['node_id'] == 'final' else PROMPTS['roles'][call['role']]
                if original['few_shot']:
                    self.assertNotIn(original['few_shot'].strip(), call['messages'][-1]['content'])

        self.config['training']['datasets'] = [dataset]
        with self.assertRaisesRegex(ValueError, '有效数据集'):
            self.service.plan(self.config, 'train')

    def test_cancel_stops_between_calls_and_blocks_concurrent_jobs(self):
        entered, release = threading.Event(), threading.Event()
        original = ModelAdapter.invoke

        def slow(adapter, messages):
            entered.set()
            release.wait(5)
            return original(adapter, messages)

        with patch.object(ModelAdapter, 'invoke', slow):
            job = self.service.submit('train', self.config)
            self.assertTrue(entered.wait(5))
            with self.assertRaisesRegex(ValueError, '已有任务'):
                self.service.submit('evaluate', self.config)
            self.service.cancel(job['id'])
            release.set()
            self.wait_for_idle()
        result = self.service.public_job(job['id'])
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(result['completed_steps'], 0)
        self.assertEqual(result['summary']['model_calls'], 1)
        self.assertEqual(self.service.checkpoints(), [])

    def test_explicit_api_keys_are_injected_without_environment_mutation_or_export(self):
        self.config['models']['agent_1'].update(backend='deepseek', model='fake-live-model', base_url='https://example.com/v1')
        self.config['evaluation'].update(datasets=['gsm8k'], limit=1)
        self.service.store.save(self.config, {'agent_1': 'private-live-stub-key'})
        with self.assertRaisesRegex(ValueError, '勾选'):
            self.service.submit('evaluate', self.config)
        environment = dict(os.environ)
        seen = []

        def factory(settings, **options):
            seen.append(options.get('api_key'))
            return ModelAdapter(FakeListChatModel(responses=['The answer is 5']), settings)

        with patch('ad_l.platform_service.create_model', side_effect=factory):
            job = self.service.submit('evaluate', self.config, allow_real_api=True)
            self.wait_for_idle()
        self.assertIn('private-live-stub-key', seen)
        self.assertEqual(dict(os.environ), environment)
        record = self.service.export(job['id'])
        self.assertEqual(record['summary']['status'], 'completed')
        self.assertEqual(record['summary']['summary']['current']['real_api_calls'], 2)
        self.assertNotIn('private-live-stub-key', json.dumps(record))

    def test_restart_marks_an_incomplete_job_interrupted(self):
        identity = 'a' * 32
        job = {'id': identity, 'status': 'running', 'name': 'test', 'kind': 'train', 'created_at': '2026-11-16T00:00:00Z',
               'progress': 0, 'total': 1, 'summary': {}}
        self.service._persist(job)
        reopened = PlatformService(self.root)
        self.assertEqual(reopened.public_job(identity)['status'], 'interrupted')

    def request(self, path, *, body=None, origin=None, host='127.0.0.1:8765', token='test-token'):
        handler_type = make_handler(self.service, 'test-token')
        handler = handler_type.__new__(handler_type)
        handler.server = SimpleNamespace(server_port=8765)
        handler.path = path
        handler.headers = Message()
        handler.headers['Host'] = host
        if origin:
            handler.headers['Origin'] = origin
        output = io.BytesIO()
        handler.wfile = output
        response = {'headers': {}}
        handler.send_response = lambda value: response.update(status=value)
        handler.send_header = lambda name, value: response['headers'].update({name: value})
        handler.end_headers = lambda: None
        if body is not None:
            raw = json.dumps(body).encode()
            handler.rfile = io.BytesIO(raw)
            handler.headers['Content-Length'] = str(len(raw))
            handler.headers['Content-Type'] = 'application/json'
            handler.headers['X-ADL-Token'] = token
            handler.do_POST()
        else:
            handler.do_GET()
        response['body'] = json.loads(output.getvalue())
        return response

    def test_http_contract_rejects_cross_origin_and_redacts_credentials(self):
        self.service.store.save(self.config, {'final': 'secret-key-in-local-store'})
        result = self.request('/api/bootstrap')
        self.assertEqual(result['status'], 200)
        self.assertNotIn('secret-key-in-local-store', json.dumps(result))
        self.assertEqual(self.request('/api/jobs', body={}, token='wrong')['status'], 403)
        self.assertEqual(self.request('/api/config', body={'config': self.config}, origin='https://example.com')['status'], 403)
        self.assertEqual(self.request('/api/bootstrap', host='evil.example:8765')['status'], 403)
        self.assertEqual(self.request('/api/config', body={'config': self.config})['status'], 200)
        plan = self.request('/api/plan', body={'config': self.config, 'kind': 'train'})
        self.assertEqual(plan['body']['model_calls'], 36)


if __name__ == '__main__':
    unittest.main()
