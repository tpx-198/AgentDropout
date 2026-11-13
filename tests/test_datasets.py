import contextlib
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from ad_l.datasets import DATASETS, Example, load_examples, load_training_mix, question_fingerprint, read_examples
from ad_l.models import create_model
from ad_l.roles import create_team
from ad_l.topology import EdgePolicy, TopologyConfig
from ad_l.training import score_example, train_step
from prepare_data import grouped_split, normalize, verify_directory

ROOT = Path(__file__).resolve().parents[1]


class DatasetTests(unittest.TestCase):
    def test_all_prepared_sources_have_disjoint_train_and_test_questions(self):
        for name in DATASETS:
            with self.subTest(dataset=name):
                train, _ = read_examples(ROOT / 'data' / name / 'train.jsonl')
                test, _ = read_examples(ROOT / 'data' / name / 'test.jsonl', expected_split='test')
                train_questions = {question_fingerprint(example.task) for example in train}
                test_questions = {question_fingerprint(example.task) for example in test}
                self.assertFalse(train_questions & test_questions)
                if name != 'gsm8k':
                    manifest = verify_directory(ROOT / 'data' / name)
                    self.assertTrue(manifest['sources'])
                    self.assertEqual(manifest['splits']['train']['records'], len(train))

    def test_balanced_pool_is_deterministic_and_includes_every_source(self):
        first, metadata = load_training_mix(ROOT / 'data', DATASETS, 2, 7)
        second, _ = load_training_mix(ROOT / 'data', DATASETS, 2, 7)
        self.assertEqual(first, second)
        self.assertEqual([example.dataset for example in first], list(DATASETS) * 2)
        self.assertEqual(metadata['selected'], 8)
        self.assertEqual(metadata['sampling']['strategy'], 'round_robin')
        self.assertTrue(all(example.split == 'train' for example in first))
        self.assertTrue(all(value['split'] == 'train' for value in metadata['datasets'].values()))

    def test_test_and_validation_data_are_rejected_for_training(self):
        for path in (ROOT / 'data/gsm8k/test.jsonl', ROOT / 'data/aqua/test.jsonl', ROOT / 'data/aqua/validation.jsonl'):
            with self.assertRaisesRegex(ValueError, 'Expected train'):
                load_examples(path, 1, 0)
        with tempfile.TemporaryDirectory() as directory:
            renamed = Path(directory) / 'train.jsonl'
            renamed.write_bytes((ROOT / 'data/gsm8k/test.jsonl').read_bytes())
            with self.assertRaisesRegex(ValueError, 'test set'):
                load_examples(renamed, 1, 0)
            row = json.loads((ROOT / 'data/aqua/test.jsonl').read_text().splitlines()[0])
            renamed.write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'split'):
                load_examples(renamed, 1, 0)

    def test_grouped_holdout_keeps_duplicates_together(self):
        rows = [{'question': f'Question {index}', 'id': str(index)} for index in range(10)]
        rows.append({'question': '  QUESTION   0 ', 'id': 'duplicate'})
        first = grouped_split(rows, 17)
        self.assertEqual(first, grouped_split(rows, 17))
        train_ids = {row['id'] for row in first['train']}
        self.assertEqual('0' in train_ids, 'duplicate' in train_ids)
        self.assertEqual(sum(len(values) for values in first.values()), len(rows))

    def test_aqua_rationale_is_excluded_and_heldout_overlap_removed(self):
        def row(question):
            return {'question': question, 'options': ['A) one', 'B) two'], 'correct': 'B',
                    'rationale': 'PRIVATE_RATIONALE_AND_LABEL'}
        content = {name: (json.dumps(row(question)) + '\n').encode()
                   for name, question in (('train', 'same'), ('validation', 'different'), ('test', 'same'))}
        splits, metadata = normalize('aqua', content, 0)
        self.assertEqual(splits['train'], [])
        self.assertEqual(metadata['excluded_train_overlap'], 1)
        self.assertNotIn('PRIVATE_RATIONALE_AND_LABEL', str(splits))

    def test_multiple_choice_scoring(self):
        example = Example('aqua:train:0', 'Question?', 'B', dataset='aqua', answer_type='choice', choices=('one', 'two'))
        for answer in ('The answer is B', 'The answer is (b).', 'Answer: B', r'\boxed{B}', 'B'):
            self.assertTrue(score_example(answer, example)['correct'], answer)
        for answer in ('The answer is A', 'The answer is E', 'A discussion of option B', 'The answer is B or A'):
            self.assertFalse(score_example(answer, example)['correct'], answer)
        self.assertTrue(score_example('The answer is -3.5', Example('x', 'Question?', '-3.50'))['correct'])
        self.assertTrue(score_example('The answer is 1/2', Example('x', 'Question?', '0.5'))['correct'])
        self.assertFalse(score_example('The answer is 1/2', Example('x', 'Question?', '1'))['correct'])
        self.assertTrue(score_example('The answer is 1e3', Example('x', 'Question?', '1000'))['correct'])

    def test_choice_rewards_train_and_labels_never_enter_model_messages(self):
        policy = EdgePolicy(TopologyConfig())
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.1)
        team = create_team()
        adapters = {agent.node_id: create_model(mock_response='The answer is B') for agent in team}
        example = Example('aqua:train:0', 'Select the correct option.\nA) one\nB) two', 'B',
                          dataset='aqua', answer_type='choice', choices=('one', 'two'))
        report = train_step(policy, optimizer, [example], adapters, torch.Generator().manual_seed(0), sparsity=0)
        self.assertEqual(report['episodes'][0]['reward'], 1)
        self.assertTrue(report['updated'])
        self.assertEqual(report['episodes'][0]['dataset'], 'aqua')
        for call in report['episodes'][0]['result']['trace']:
            self.assertIn('option letter', call['messages'][0]['content'])
            self.assertNotIn('"target"', str(call['messages']))
        with self.assertRaisesRegex(ValueError, 'training examples'):
            train_step(policy, optimizer, [replace(example, split='test')], adapters, torch.Generator())

    def test_manifest_detects_edited_data(self):
        with tempfile.TemporaryDirectory() as directory:
            source = ROOT / 'data/svamp'
            target = Path(directory)
            for name in ('train.jsonl', 'manifest.json'):
                (target / name).write_bytes((source / name).read_bytes())
            with (target / 'train.jsonl').open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'manifest'):
                load_examples(target / 'train.jsonl', 1, 0)

    def test_single_dataset_plan_and_legacy_data_option(self):
        import sys
        spec = importlib.util.spec_from_file_location('train_data_entry', ROOT / 'train.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for options in (['--datasets', 'aqua'], ['--data', str(ROOT / 'data/gsm8k/train.jsonl')]):
            output = io.StringIO()
            with patch.object(sys, 'argv', ['train.py', '--plan', *options]), \
                    patch.object(module, 'adapters_from_args', side_effect=AssertionError('model created')), \
                    contextlib.redirect_stdout(output):
                module.main()
            self.assertEqual(len(json.loads(output.getvalue())['sample_ids']), 6)


if __name__ == '__main__':
    unittest.main()
