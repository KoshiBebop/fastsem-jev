"""In-process JevBench adapter; JevBench itself is an optional checkout dependency."""
import math
import time

from .engine import MODEL, REVISION


class FastSemLocalAdapter:
    name = "fastsem_local"
    cost_basis = "self_hosted_gpu_cost_not_estimated"

    def __init__(self, endpoint=None, model=None, key_env="", timeout_s=None,
                 price_input_per_m=None, price_output_per_m=None, revision=None,
                 layer=16, retain_ratio=0.25):
        self.endpoint = endpoint or MODEL
        self.model = model or self.endpoint
        self.revision = revision or REVISION
        self.layer, self.retain_ratio = layer, retain_ratio
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m
        self._engine = None

    def load(self):
        if self._engine is None:
            from .engine import FastSemJev
            started = time.perf_counter()
            self._engine = FastSemJev(self.endpoint, self.revision, self.layer, self.retain_ratio)
            self.load_s = time.perf_counter() - started
        return self._engine

    def build_request(self, task):
        labels = task.labels
        if (not isinstance(labels, list) or not 2 <= len(labels) <= 16
                or any(not isinstance(k, str) or not k for k in labels)
                or len(set(labels)) != len(labels)):
            raise ValueError("Expected 2-16 unique, nonempty string labels")
        kind = task.question["type"]
        criteria = task.question.get("criteria")
        if kind == "noul":
            if set(labels) != {"yes", "no"}:
                raise ValueError("Noul labels must be yes and no")
            if criteria is not None and not isinstance(criteria, dict):
                raise ValueError("Noul criteria must be an object or null")
            criteria = criteria or {}
            descriptions = {"yes": criteria.get("true", "The proposition is true."),
                            "no": criteria.get("false", "The proposition is false.")}
        elif kind == "choice":
            if not isinstance(criteria, dict) or set(criteria) != set(labels):
                raise ValueError("Choice criteria must match the declared labels")
            descriptions = criteria
        elif kind == "score":
            if not isinstance(criteria, list) or labels != [str(i) for i in range(len(criteria))]:
                raise ValueError("Score labels must follow the ordered level indices")
            descriptions = dict(zip(labels, criteria))
        else:
            raise ValueError("Unsupported question type")
        options = [{"id": k, "description": f"{k}: {descriptions[k]}"} for k in labels]
        row = {"id": task.id, "state": task.state,
               "question": task.question["instructions"], "options": options}
        from ._core import validate_row
        validate_row(row)
        return row

    def run(self, task):
        from jevbench.adapters.base import DecisionResult
        result = DecisionResult(adapter=self.name, ok=False, model=self.model, probs_source="native")
        started = time.perf_counter()
        try:
            row = self.build_request(task)
            result.request_body = row
            engine = self.load()
            answer = engine.decide(row["state"], row["question"], row["options"])
            probs = answer["probabilities"]
            if not isinstance(probs, dict) or set(probs) != set(task.labels):
                raise RuntimeError("Model probability labels do not match the task")
            if any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or not 0 <= v <= 1 for v in probs.values()):
                raise RuntimeError("Model probabilities must be finite numbers in [0, 1]")
            # The shared scorer decides sum tolerance; never fill or renormalize here.
            result.probs = probs
            result.usage = {"input_tokens": answer["diagnostics"]["input_tokens"], "output_tokens": 0}
            result.raw = {"answer": answer, "runtime": {
                **engine.metadata, "configuration": engine.spec,
                "probability_origin": "native-option-logit-softmax"}}
            result.ok = True
        except Exception as error:
            result.error = f"{type(error).__name__}: {error}"
            if isinstance(error, ValueError):
                result.status = 422
        result.latency_s = time.perf_counter() - started
        return result

    def reserve_estimate(self, task):
        return 0.0
