import json
import inspect
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fastsem_jev.typesafe_server import create_server
from fastsem_jev.engine import FastSemJev


class FakeEngine:
    def decide(self, state, question, options):
        labels = [option["id"] for option in options]
        if labels == ["yes", "no"]:
            probabilities = {"yes": 0.8, "no": 0.2}
        elif labels == ["0", "1", "2"]:
            probabilities = {"0": 0.1, "1": 0.6, "2": 0.3}
        else:
            probabilities = {label: (0.75 if label == "approve" else 0.25 / (len(labels) - 1))
                             for label in labels}
        return {"prediction": max(probabilities, key=probabilities.get),
                "probabilities": probabilities,
                "diagnostics": {"input_tokens": 12}}


class TypeSafeServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server(FakeEngine(), "fastsem-test", "127.0.0.1", 0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def post(self, payload):
        request = Request(self.base + "/v1/systemone", json.dumps(payload).encode(),
                          {"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=2) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            return error.code, json.loads(error.read())

    def test_health(self):
        with urlopen(self.base + "/health", timeout=2) as response:
            self.assertEqual(json.loads(response.read()), {"status": "ok", "model": "fastsem-test"})

    def test_primary_method_default_is_l16r25(self):
        self.assertEqual(inspect.signature(FastSemJev).parameters["retain_ratio"].default, 0.25)

    def test_noul_choice_and_score_shapes(self):
        status, payload = self.post({
            "state": "A sample state",
            "model": "fastsem-test",
            "questions": {
                "binary": {"type": "noul", "instructions": "Is it true?",
                           "criteria": {"true": "The claim holds.", "false": "The claim fails."}},
                "route": {"type": "choice", "instructions": "Where should it go?",
                          "criteria": {"approve": "Approve it", "reject": "Reject it"}},
                "urgency": {"type": "score", "instructions": "How urgent?",
                            "criteria": ["low", "medium", "high"]},
            },
        })
        self.assertEqual(status, 200)
        answers = payload["answers"]
        self.assertEqual(answers["binary"]["noul"], 0.8)
        self.assertEqual(answers["route"]["choice"], "approve")
        self.assertEqual(set(answers["route"]["probabilities"]), {"approve", "reject"})
        self.assertAlmostEqual(answers["urgency"]["score"], 1.2)
        self.assertEqual(payload["usage"], {"input_tokens": 36, "output_tokens": 0})

    def test_reject_unknown_model_and_bad_question(self):
        status, payload = self.post({"state": "text", "model": "someone-else",
                                     "questions": {"q": {"type": "noul", "instructions": "?"}}})
        self.assertEqual(status, 400)
        self.assertIn("only serves", payload["error"])
        status, _ = self.post({"state": "text", "questions": {"q": {"type": "freeform"}}})
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
