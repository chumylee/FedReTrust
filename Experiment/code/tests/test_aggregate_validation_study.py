"""The paired pilot must stop when a local training replay changes."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggregate_validation_study import assert_exact_replay


class PairedReplayTests(unittest.TestCase):
    def setUp(self):
        self.metrics = {"shuffle_seed": 1234, "optimizer_steps": 8,
                        "online_train_loss": .123456}
        self.expected = {**self.metrics, "submitted_state_sha256": "original",
                         "examples": 512}

    def test_exact_replay_is_accepted(self):
        assert_exact_replay("original", self.metrics, 512, self.expected)

    def test_changed_state_data_or_optimizer_evidence_is_rejected(self):
        changes = {"submitted_state_sha256": "changed", "examples": 511,
                   "shuffle_seed": 1235, "optimizer_steps": 7,
                   "online_train_loss": .123457}
        for name, value in changes.items():
            with self.subTest(name=name), self.assertRaises(AssertionError):
                assert_exact_replay("original", self.metrics, 512,
                                    {**self.expected, name: value})


if __name__ == "__main__":
    unittest.main()
