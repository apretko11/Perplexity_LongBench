"""Offline tests for driver, with mocked HF imports/model loading.

Use: python -m unittest -v test_benchmark_celerity_longbench_v2
"""
import importlib.util
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

# Load this driver with temporary mocked dependencies. Scope all sys.modules
# edits to the import itself so the existing Step 1–3 tests retain the real
# transformers/datasets/longbench_perplexity imports during test discovery.
from unittest.mock import patch as mock_patch

reference = types.ModuleType('longbench_perplexity')
reference.DEFAULT_LONGBENCH_TASKS = ['narrativeqa', 'qasper']
reference.make_longbench_prompt = lambda example, task: example['context'] + '\n' + example['input']
reference.tokenize_prompt = lambda tok, prompt, use_chat_template=False: [1, 2, 3]
called = []

def reference_score(*, model, tokenizer, prompt, answer, max_length, stride, use_chat_template):
    called.append((prompt, answer, max_length, stride, use_chat_template))
    return float(len(answer)), len(answer)

reference.answer_conditional_ppl = reference_score
fake_ds = types.ModuleType('datasets')
fake_ds.load_dataset = lambda *args, **kwargs: None
fake_tf = types.ModuleType('transformers')
fake_tf.AutoTokenizer = object
fake_tf.AutoModelForCausalLM = object
mocked = {'datasets':fake_ds, 'transformers':fake_tf, 'longbench_perplexity':reference}
script = Path(__file__).with_name('benchmark_celerity_longbench_v2.py')
# When the test file lives in tests/ and the driver is in the repo root.
if not script.exists():
    script = Path(__file__).resolve().parent.parent / script.name
spec = importlib.util.spec_from_file_location('pilot_benchmark_under_test', script)
pilot = importlib.util.module_from_spec(spec)
with mock_patch.dict(sys.modules, mocked):
    spec.loader.exec_module(pilot)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text)))}


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __iter__(self):
        return iter(self.rows)
    def __getitem__(self, i):
        return self.rows[i]


class BatchTests(unittest.TestCase):
    def test_context_parser(self):
        self.assertEqual(pilot.named_context('adpretko/celerity-271m-4k-ad0p0'), 4096)
        self.assertEqual(pilot.named_context('adpretko/celerity-504m-16k-ad0p0'), 16384)
        self.assertIsNone(pilot.named_context('adpretko/nope'))

    def test_reference_scorer_invoked(self):
        called.clear()
        example = {'task':'narrativeqa','example_index':7,'example_id':'xyz',
                   'prompt':'Q', 'answer':'Paris', 'length_group':'4k'}
        row = pilot.score_one(None, FakeTokenizer(), example, context_limit=4096, stride=64)
        self.assertEqual(called, [('Q','Paris',4096,64,False)])
        self.assertEqual(row['tokens_scored'], 5)
        self.assertAlmostEqual(row['avg_nll'], 1.0)
        self.assertAlmostEqual(row['perplexity'], math.e)

    def test_manifest_requires_identical_settings(self):
        args = SimpleNamespace(order='recent',model_offset=0,limit_models=10,
            tasks=['narrativeqa'],examples_per_task=5,seed=42,sampling='random',
            context_mode='common',common_context=4096,answer_stride=64,dtype='float16')
        doc = {'settings': pilot.settings(args), 'models':[]}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'manifest.json'
            pilot.atomic_json(path,doc)
            self.assertEqual(pilot.freeze_manifest(path,None,args), doc)
            args.seed += 1
            with self.assertRaisesRegex(ValueError, 'different settings'):
                pilot.freeze_manifest(path,None,args)

    def test_dataset_sample_frozen(self):
        rows = [dict(_id=str(i),context='text',input='Q',answers=[str(i + 1)],length='4k')
                for i in range(12)]
        data = FakeDataset(rows)
        cfg = dict(tasks=['narrativeqa','qasper'],examples_per_task=4,
                   sampling='random',seed=71)
        doc = dict(settings=cfg, longbench_revision='frozen', sample_indices={}, sample_id_digest={})
        with tempfile.TemporaryDirectory() as td, patch.object(pilot,'load_dataset',return_value=data):
            path = Path(td) / 'manifest.json'
            cases = pilot.select_cases(doc,path)
            self.assertEqual(len(cases),8)
            self.assertEqual([r['example_index'] for r in cases[:4]],
                             doc['sample_indices']['narrativeqa'])
            self.assertEqual([r['example_index'] for r in cases],
                             [r['example_index'] for r in pilot.select_cases(doc,path)])
            data.rows[doc['sample_indices']['narrativeqa'][0]]['answers']=['changed']
            with self.assertRaisesRegex(RuntimeError,'changed'):
                pilot.select_cases(doc,path)

    def test_run_one_saves_aggregates_and_resumes(self):
        args = SimpleNamespace(context_mode='common', common_context=4096,
                               answer_stride=64, device='cpu', dtype='float32', progress_every=10)
        spec_model = dict(model='adpretko/celerity-271m-4k-ad0p0',revision='a'*40,
                          trained_context_tokens=4096)
        cases = [dict(task='narrativeqa',example_index=i,example_id=str(i),prompt='Question',
                      answer='Paris',length_group='4k') for i in range(2)]
        with tempfile.TemporaryDirectory() as td, patch.object(pilot,'load_model',return_value=(object(),FakeTokenizer(),'tokdigest')):
            path=Path(td)
            row=pilot.run_one(spec_model,cases,args,path)
            self.assertEqual(row['status'],'ok',row)
            self.assertEqual(row['example_count'],2)
            self.assertEqual(row['scored_tokens'],10)
            self.assertAlmostEqual(row['perplexity'],math.e)
            pilot.save_summary(path,row)
            selected = {'models':[spec_model]}
            self.assertEqual(pilot.completed_models(path,selected), {spec_model['model']})
            self.assertEqual(len((path/'summary.csv').read_text().splitlines()),2)
            self.assertEqual(len(list((path/'examples').glob('*.jsonl'))),1)
            self.assertEqual(len(list((path/'tasks').glob('*.jsonl'))),1)


if __name__ == '__main__':
    unittest.main()
