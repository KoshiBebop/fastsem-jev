import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastsem_jev.jevbench_adapter import FastSemLocalAdapter


def task(kind="choice", labels=None, criteria=None):
    return SimpleNamespace(id="t", state="evidence", expected="secret-answer",
        labels=labels or ["allow", "deny"], question={"type": kind,
        "instructions": "criterion", "criteria": criteria})


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = FastSemLocalAdapter()

    def test_three_types_and_declared_order(self):
        examples = [
            (task(criteria={"deny": "D", "allow": "A"}), ["allow: A", "deny: D"]),
            (task("noul", ["no", "yes"], {"true": "T", "false": "F"}), ["no: F", "yes: T"]),
            (task("score", ["0", "1", "2"], ["low", "mid", "high"]), ["0: low", "1: mid", "2: high"]),
        ]
        for t, descriptions in examples:
            row = self.adapter.build_request(t)
            self.assertEqual([o["id"] for o in row["options"]], t.labels)
            self.assertEqual([o["description"] for o in row["options"]], descriptions)
            self.assertNotIn("expected", row)

    def test_load_once(self):
        with patch("fastsem_jev.engine.FastSemJev") as factory:
            self.assertIs(self.adapter.load(), self.adapter.load())
            factory.assert_called_once()

    def test_exact_probabilities_and_failures(self):
        try:
            from jevbench.adapters.base import DecisionResult
        except ImportError:
            self.skipTest("Set PYTHONPATH to the JevBench checkout for integration tests")
        engine = MagicMock(metadata={})
        engine.spec = {}
        self.adapter._engine = engine
        t = task(criteria={"allow": "A", "deny": "D"})
        # Deliberately non-normalized: the official scorer owns sum validation.
        valid = {"allow": 0.2, "deny": 0.7}
        engine.decide.return_value = {"probabilities": valid, "diagnostics": {"input_tokens": 10}}
        result = self.adapter.run(t)
        self.assertIsInstance(result, DecisionResult)
        self.assertTrue(result.ok)
        self.assertIs(result.probs, valid)
        self.assertEqual(result.usage["output_tokens"], 0)
        for bad in ({"allow": 1.0}, {"allow": 0.5, "deny": 0.5, "extra": 0.0},
                    {"allow": math.nan, "deny": 0.5}, {"allow": math.inf, "deny": 0.0},
                    {"allow": -0.1, "deny": 1.1}, {"allow": True, "deny": 0.0},
                    {"allow": "0.5", "deny": 0.5}):
            engine.decide.return_value["probabilities"] = bad
            result = self.adapter.run(t)
            self.assertFalse(result.ok)
            self.assertIsNone(result.probs)
        engine.decide.side_effect = RuntimeError("device failure")
        self.assertFalse(self.adapter.run(t).ok)

    def test_malformed_question(self):
        for t in (task(criteria={"allow": "A"}), task("score", ["0", "2"], ["low", "high"]),
                  task("noul", ["a", "b"], {}), task("freeform", criteria={})):
            with self.assertRaises(ValueError):
                self.adapter.build_request(t)


if __name__ == "__main__":
    unittest.main()
