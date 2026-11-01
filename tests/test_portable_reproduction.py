"""Guard meaningful failure modes of the new saved-artifact entry point."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from common import new_output
from reproduce_results import calculate, V6, read


class ReproductionGuards(unittest.TestCase):
    def setUp(self):
        self.rows = [r for r in read('block_metrics') if r['method'] in {V6, 'RFA'}
                     and r['scenario'] == 'label_flip' and r['metric'] == 'overall_mean_5_24_pp'
                     and r['regime'] == 'non-IID']

    def test_incomplete_block_is_not_silently_averaged(self):
        with self.assertRaisesRegex(ValueError, 'five'):
            calculate(self.rows[:-1])

    def test_duplicate_block_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            calculate(self.rows + [self.rows[0]])

    def test_different_training_seed_cannot_be_paired(self):
        changed = [dict(r) for r in self.rows]
        changed[0]['seed'] = str(int(changed[0]['seed']) + 100)
        with self.assertRaisesRegex(ValueError, 'seeds differ'):
            calculate(changed)

    def test_nonfinite_measurement_is_rejected(self):
        changed = [dict(r) for r in self.rows]
        changed[0]['value'] = 'nan'
        with self.assertRaisesRegex(ValueError, 'finite'):
            calculate(changed)

    def test_output_label_cannot_target_evidence(self):
        with self.assertRaises(ValueError):
            new_output('reproduction', '../Experiment/outputs')


if __name__ == '__main__':
    unittest.main()
