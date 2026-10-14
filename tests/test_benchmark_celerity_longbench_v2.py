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


    def test_ephemeral_cache_forwarded_and_removed_after_success(self):
        # No touching a shared Hugging Face cache; a unique owned directory
        # is passed to model loading and removed when the model finishes.
        args = SimpleNamespace(context_mode='common', common_context=4096,
            answer_stride=64, device='cpu', dtype='float32', progress_every=10,
            ephemeral_model_cache=True)
        spec_model = dict(model='adpretko/celerity-271m-4k-ad0p0',
            revision='a'*40, trained_context_tokens=4096)
        cases = [dict(task='narrativeqa', example_index=0, example_id='ex',
            prompt='Question', answer='Paris', length_group='4k')]
        seen = []

        def fake_load(spec, dtype, device, cache_dir=None):
            self.assertIsNotNone(cache_dir)
            path = Path(cache_dir)
            self.assertTrue(path.is_dir())
            (path / 'model-weights.bin').write_bytes(b'dummy')
            seen.append(path)
            return object(), FakeTokenizer(), 'digest'

        with tempfile.TemporaryDirectory() as root:
            args.model_cache_parent = Path(root) / 'ephemeral'
            untouched = Path(root) / 'shared_cache'
            untouched.mkdir()
            (untouched / 'existing.bin').write_bytes(b'keep')
            with patch.object(pilot, 'load_model', side_effect=fake_load):
                row = pilot.run_one(spec_model, cases, args, Path(root)/'results')
            self.assertEqual(row['status'], 'ok', row)
            self.assertEqual(len(seen), 1)
            self.assertFalse(seen[0].exists())
            self.assertTrue((untouched/'existing.bin').exists())
            self.assertEqual(list(args.model_cache_parent.iterdir()), [])

    def test_ephemeral_cache_removed_when_loading_fails(self):
        args = SimpleNamespace(context_mode='common', common_context=4096,
            answer_stride=64, device='cpu', dtype='float32', progress_every=10,
            ephemeral_model_cache=True)
        spec_model = dict(model='adpretko/celerity-271m-4k-ad0p0',
            revision='a'*40, trained_context_tokens=4096)
        seen = []

        def broken_load(spec, dtype, device, cache_dir=None):
            path = Path(cache_dir)
            (path/'partial-download.bin').write_bytes(b'part')
            seen.append(path)
            raise RuntimeError('simulated failure')

        with tempfile.TemporaryDirectory() as root:
            args.model_cache_parent = Path(root)/'ephemeral'
            with patch.object(pilot, 'load_model', side_effect=broken_load):
                row = pilot.run_one(spec_model, [], args, Path(root)/'results')
            self.assertEqual(row['status'], 'error')
            self.assertIn('simulated failure', row['error'])
            self.assertEqual(len(seen), 1)
            self.assertFalse(seen[0].exists())
            self.assertEqual(list(args.model_cache_parent.iterdir()), [])

    def test_load_model_scoped_cache_applies_to_model_and_tokenizer(self):
        class ToyLM:
            config = SimpleNamespace(model_type='celerity')
            def to(self, device): return self
            def eval(self): return self
        class ToyTokenizer:
            special_tokens_map = {'bos_token': '<s>'}
            def get_vocab(self): return {'hello': 0}
        class CaptureFactory:
            def __init__(self, return_value):
                self.return_value = return_value
                self.kwargs = None
            def from_pretrained(self, *a, **kw):
                self.kwargs = kw
                return self.return_value
        tf = CaptureFactory(ToyTokenizer())
        mf = CaptureFactory(ToyLM())
        spec_model = dict(model='adpretko/celerity-271m-4k-ad0p0',revision='a'*40)
        with patch.object(pilot,'AutoTokenizer',tf), patch.object(pilot,'AutoModelForCausalLM',mf):
            _,_,digest=pilot.load_model(spec_model,'float32','cpu',cache_dir='/temp/only-this-model')
        self.assertEqual(tf.kwargs['cache_dir'],'/temp/only-this-model')
        self.assertEqual(mf.kwargs['cache_dir'],'/temp/only-this-model')
        self.assertTrue(digest)


    def test_frozen_8k_experiment_reuses_4k_revisions_and_examples(self):
        rows = [
            dict(_id='id0', context='alpha', input='Q', answers=['first'], length='4k'),
            dict(_id='id1', context='beta', input='Q', answers=['second'], length='4k'),
            dict(_id='id2', context='gamma', input='Q', answers=['third'], length='4k'),
        ]
        cases_digest = pilot.sha_json([
            (str(rows[i]['_id']), str(rows[i]['answers'][0])) for i in [0, 2]
        ])
        specs = [
            dict(model=f'adpretko/celerity-271m-{size}k-ad0p0',
                 revision=str(size) * 40, trained_context_tokens=size * 1024)
            for size in (4, 8, 16)
        ]
        baseline_args = SimpleNamespace(
            order='recent', model_offset=0, limit_models=0,
            tasks=['narrativeqa'], examples_per_task=2, seed=42,
            sampling='random', context_mode='common', common_context=4096,
            answer_stride=128, dtype='bfloat16',
        )
        baseline = {
            'version': 2, 'settings': pilot.settings(baseline_args),
            'models': specs, 'total_celerity_repositories': 3,
            'excluded_models': [], 'longbench_revision': 'frozen-dataset',
            'sample_indices': {'narrativeqa': [0, 2]},
            'sample_id_digest': {'narrativeqa': cases_digest},
        }
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / '4k_manifest.json'
            dst = Path(td) / '8k_manifest.json'
            pilot.atomic_json(src, baseline)
            args = SimpleNamespace(**vars(baseline_args))
            args.common_context = 8192
            args.source_manifest = src
            args.min_trained_context = 8192
            args.max_run_models = 1  # A pilot limit must not change frozen settings.
            frozen_pilot_settings = pilot.settings(args)
            args.max_run_models = 0
            self.assertEqual(frozen_pilot_settings, pilot.settings(args))
            with patch.object(pilot, 'discover', side_effect=AssertionError('must not rediscover HF')):
                doc = pilot.freeze_manifest(dst, None, args)
            self.assertEqual([m['trained_context_tokens'] for m in doc['models']], [8192, 16384])
            self.assertEqual([m['revision'] for m in doc['models']], [str(8)*40, str(16)*40])
            self.assertEqual(doc['sample_indices'], baseline['sample_indices'])
            self.assertEqual(doc['sample_id_digest'], baseline['sample_id_digest'])
            self.assertEqual(doc['longbench_revision'], baseline['longbench_revision'])
            self.assertEqual(len(doc['models']), 2)
            self.assertEqual(pilot.freeze_manifest(dst, None, args)['models'], doc['models'])
            with patch.object(pilot, 'load_dataset', return_value=FakeDataset(rows)):
                cases = pilot.select_cases(doc, dst)
            self.assertEqual([c['example_index'] for c in cases], [0, 2])
            self.assertEqual([c['answer'] for c in cases], ['first', 'third'])
            baseline['extra'] = 'tampered'
            pilot.atomic_json(src, baseline)
            with self.assertRaisesRegex(ValueError, 'different settings'):
                pilot.freeze_manifest(dst, None, args)

    def test_frozen_8k_experiment_rejects_changed_scoring_policy(self):
        baseline_args = SimpleNamespace(
            order='recent', model_offset=0, limit_models=0,
            tasks=['narrativeqa'], examples_per_task=1, seed=42,
            sampling='random', context_mode='common', common_context=4096,
            answer_stride=128, dtype='bfloat16',
        )
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / 'source.json'
            doc = dict(
                version=2, settings=pilot.settings(baseline_args),
                models=[dict(model='adpretko/celerity-504m-8k-ad0p0',
                             revision='f'*40, trained_context_tokens=8192)],
                total_celerity_repositories=1, excluded_models=[],
                longbench_revision='frozen',
                sample_indices={'narrativeqa': [0]},
                sample_id_digest={'narrativeqa': 'a'*64},
            )
            pilot.atomic_json(src, doc)
            args = SimpleNamespace(**vars(baseline_args))
            args.source_manifest = src
            args.min_trained_context = 8192
            args.common_context = 8192
            args.dtype = 'float16'
            with self.assertRaisesRegex(ValueError, 'dtype'):
                pilot.freeze_manifest(Path(td) / 'new.json', None, args)


if __name__ == '__main__':
    unittest.main()
