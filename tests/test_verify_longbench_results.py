"""Minimal offline audit tests; no model weights or dependencies."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from verify_longbench_results import verify_gold, verify_generated


def write(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')


def score(nll, tokens, **extra):
    return dict(nll_sum=nll, tokens_scored=tokens, avg_nll=nll / tokens if tokens else None,
                perplexity=math.exp(nll / tokens) if tokens else None, **extra)


class AuditorTests(unittest.TestCase):
    def test_gold_first_and_aggregation(self):
        with tempfile.TemporaryDirectory() as td:
            summary = Path(td) / 'gold.jsonl'
            detail = Path(td) / 'gold_examples.jsonl'
            rows = [
                dict(row_type='example', task='narrativeqa', example_index=0, mode='answer',
                     reference_count_scored=1, references=[score(4., 2)],
                     reference_macro_avg_nll=2., **score(4., 2)),
                dict(row_type='example', task='narrativeqa', example_index=1, mode='answer',
                     reference_count_scored=1, references=[score(6., 1)],
                     reference_macro_avg_nll=6., **score(6., 1)),
            ]
            tot = 10. / 3
            sums = [dict(task='narrativeqa', examples=2, tokens_scored=3, avg_nll=tot,
                         perplexity=math.exp(tot), example_macro_avg_nll=4.),
                    dict(task='OVERALL', examples=2, tokens_scored=3, avg_nll=tot,
                         perplexity=math.exp(tot), example_macro_avg_nll=4., macro_avg_nll=tot)]
            write(detail, rows); write(summary, sums)
            result = verify_gold(summary)
            self.assertEqual(result['tokens'], 3)
            sums[1]['tokens_scored'] = 4
            write(summary, sums)
            with self.assertRaisesRegex(ValueError, 'count mismatch'):
                verify_gold(summary)

    def test_generated_traced(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'generated.jsonl'
            ex = dict(row_type='example', task='narrativeqa', example_index=0,
                      generated_tokens=3, verified_generation_chunks=2,
                      scoring_context_policy='head_tail_prompt_recent_generation_v1', **score(3., 3))
            summ = dict(row_type='task_summary', task='narrativeqa', examples=1,
                        tokens_scored=3, avg_nll=1., perplexity=math.e)
            overall = dict(row_type='overall_summary', task='OVERALL', examples=1,
                           tokens_scored=3, avg_nll=1., perplexity=math.e,
                           scoring_context_policy='head_tail_prompt_recent_generation_v1')
            write(path, [ex, summ, overall])
            self.assertEqual(verify_generated(path)['tokens'], 3)
            ex['generated_tokens'] = 4
            write(path, [ex, summ, overall])
            with self.assertRaisesRegex(ValueError, 'token count'):
                verify_generated(path)


if __name__ == '__main__':
    unittest.main()
