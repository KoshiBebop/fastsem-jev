"""Minimal TypeSafe-compatible HTTP service for JevBench adapters."""

from __future__ import annotations

import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


MAX_REQUEST_BYTES = 16 * 1024 * 1024


def _options(question: dict[str, Any]) -> list[dict[str, str]]:
    qtype = question.get("type")
    criteria = question.get("criteria")
    if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
        raise ValueError("question instructions must be a non-empty string")

    if qtype == "noul":
        criteria = criteria if isinstance(criteria, dict) else {}
        yes = criteria.get("yes", criteria.get("true", "The proposition is true."))
        no = criteria.get("no", criteria.get("false", "The proposition is false."))
        options = [{"id": "yes", "description": f"yes: {yes}"},
                   {"id": "no", "description": f"no: {no}"}]
    elif qtype == "choice":
        if not isinstance(criteria, dict):
            raise ValueError("choice criteria must be an object mapping labels to descriptions")
        options = [{"id": str(label), "description": f"{label}: {description or label}"}
                   for label, description in criteria.items()]
    elif qtype == "score":
        if not isinstance(criteria, list):
            raise ValueError("score criteria must be an ordered list of levels")
        options = [{"id": str(index), "description": f"{index}: {description or index}"}
                   for index, description in enumerate(criteria)]
    else:
        raise ValueError("question type must be one of: noul, choice, score")

    if not 2 <= len(options) <= 16:
        raise ValueError("fastsem-jev supports 2 to 16 options per question")
    if len({option["id"] for option in options}) != len(options):
        raise ValueError("question labels must be unique")
    return options


def _answer(question: dict[str, Any], probabilities: dict[str, float]) -> dict[str, Any]:
    qtype = question["type"]
    confidence = max(probabilities.values())
    answer: dict[str, Any] = {"type": qtype, "confidence": confidence}
    if qtype == "noul":
        answer["noul"] = probabilities["yes"]
    elif qtype == "choice":
        answer["choice"] = max(sorted(probabilities), key=probabilities.__getitem__)
        answer["probabilities"] = probabilities
    else:
        answer["score"] = sum(int(label) * probability
                              for label, probability in probabilities.items())
        answer["probabilities"] = probabilities
    return answer


def create_server(engine, model_name: str = "fastsem-jev", host: str = "127.0.0.1", port: int = 8000):
    """Create a serial HTTP server; serial requests match JevBench's runner."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "fastsem-jev/0.1"

        def _send(self, status: int, payload: dict[str, Any]):
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                self._send(404, {"error": "not found"})
                return
            self._send(200, {"status": "ok", "model": model_name})

        def do_POST(self):
            if self.path != "/v1/systemone":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError(f"request body must be between 1 and {MAX_REQUEST_BYTES} bytes")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict) or not isinstance(body.get("questions"), dict):
                    raise ValueError("request must contain a questions object")
                if not body["questions"]:
                    raise ValueError("questions must not be empty")
                if "model" in body and body["model"] not in (None, "", model_name):
                    raise ValueError(f"this server only serves model {model_name!r}")
                state = body.get("state")
                if not isinstance(state, (str, dict, list)) or not state:
                    raise ValueError("state must be a non-empty string, object, or array")

                answers = {}
                input_tokens = 0
                for question_id, question in body["questions"].items():
                    if not isinstance(question_id, str) or not isinstance(question, dict):
                        raise ValueError("each question must map a string id to an object")
                    options = _options(question)
                    result = engine.decide(state, question["instructions"], options)
                    probabilities = result["probabilities"]
                    labels = {option["id"] for option in options}
                    if not isinstance(probabilities, dict) or set(probabilities) != labels:
                        raise RuntimeError(f"model returned an invalid label set for question {question_id!r}")
                    if any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(value) or not 0.0 <= value <= 1.0
                           for value in probabilities.values()):
                        raise RuntimeError(f"model returned invalid probabilities for question {question_id!r}")
                    if abs(sum(probabilities.values()) - 1.0) > 1e-3:
                        raise RuntimeError(f"model probabilities do not sum to one for question {question_id!r}")
                    if result.get("diagnostics", {}).get("input_tokens"):
                        input_tokens += int(result["diagnostics"]["input_tokens"])
                    answers[question_id] = _answer(question, probabilities)
                self._send(200, {"model": model_name, "answers": answers,
                                 "usage": {"input_tokens": input_tokens, "output_tokens": 0}})
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                self._send(400, {"error": str(error)})
            except Exception as error:  # surface model/runtime failures as server errors
                self._send(500, {"error": f"inference failed: {type(error).__name__}"})

        def log_message(self, format, *args):
            # Avoid printing request payloads or benchmark text to server logs.
            super().log_message(format, *args)

    return HTTPServer((host, port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--retain-ratio", type=float, default=0.25)
    parser.add_argument("--served-name", default="fastsem-jev")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    from .engine import FastSemJev

    engine = FastSemJev(args.model, args.revision, args.layer, args.retain_ratio)
    server = create_server(engine, args.served_name, args.host, args.port)
    print(f"fastsem-jev serving {args.served_name} at http://{args.host}:{args.port}", flush=True)
    print(f"config: l{args.layer}r{args.retain_ratio * 100:g}; POST /v1/systemone; GET /health", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down fastsem-jev.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
