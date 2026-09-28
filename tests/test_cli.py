import contextlib
import io
import json
import unittest
from unittest.mock import MagicMock, patch

from fastsem_jev.cli import main


class CliTests(unittest.TestCase):
    def test_inline_request_runs_once_and_stdout_is_json(self):
        engine = MagicMock(device="cuda:0", metadata={})
        engine.spec = {}
        engine.decide.return_value = {"prediction": "no", "probabilities": {"yes": 0.1, "no": 0.9}}
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("fastsem_jev.cli.FastSemJev", return_value=engine), \
             patch("fastsem_jev.cli.torch.cuda.synchronize"), \
             patch("fastsem_jev.cli.torch.cuda.get_device_name", return_value="test"), \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            main(["--state", "Arrives tomorrow", "--question", "Delivered?",
                  "--options", "yes=Arrived", "no=Not arrived"])
        engine.decide.assert_called_once_with("Arrives tomorrow", "Delivered?",
            [{"id": "yes", "description": "Arrived"}, {"id": "no", "description": "Not arrived"}])
        self.assertEqual(json.loads(stdout.getvalue())["prediction"], "no")
        self.assertEqual(json.loads(stderr.getvalue())["summary"]["timed_calls"], 1)

    def test_bad_requests_fail_before_model_load(self):
        for args in (["--state", "text"],
                     ["--state", "text", "--question", "?", "--options", "yes", "yes"],
                     ["--input", "missing.json", "--question", "?"],
                     ["--state", "text", "--question", "?", "--options", "yes"]):
            with self.subTest(args=args), patch("fastsem_jev.cli.FastSemJev") as load, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
