"""Public entry point for the evaluated single-stage compression path."""
import re

import torch
import torch.nn.functional as F

from ._core import direct_messages, load_causal_model
from ._funnel import config, encode_with_evidence_mask, scheduled_logits, validate

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


class FastSemJev:
    def __init__(self, model=MODEL, revision=REVISION, layer=16, retain_ratio=0.175):
        self.spec = config("fastsem-jev", [(layer, retain_ratio)])
        validate(self.spec, 32)
        self.model, self.tokenizer, self.metadata = load_causal_model(model, revision)
        self.model.requires_grad_(False)
        validate(self.spec, len(self.model.model.layers))
        self.device = next(self.model.parameters()).device

    @torch.inference_mode()
    def decide(self, state, question, options, method="fastsem"):
        """Return option IDs/probabilities. Labels and dataset metadata are not inputs."""
        request = dict(id="inference", state=state, question=question, options=options)
        labels = [o["id"] for o in options]
        if method == "qwen_generate":
            prompt = self.tokenizer.apply_chat_template(
                direct_messages(request), tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
            self._check_length(ids)
            output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                         do_sample=False, max_new_tokens=32, use_cache=True)
            text = self.tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
            match = re.fullmatch(r"([A-Z])[.)]?", text.strip())
            index = ord(match[1]) - ord("A") if match else -1
            return {"prediction": labels[index] if 0 <= index < len(labels) else "__invalid_generation__",
                    "probabilities": None, "generated_text": text}
        if method not in {"fastsem", "semif"}:
            raise ValueError("method must be fastsem, semif, or qwen_generate")
        ids, slots, mask, _ = encode_with_evidence_mask(self.tokenizer, request, "direct")
        ids = ids.to(self.device)
        self._check_length(ids)
        weights = self.model.lm_head.weight[slots]
        if method == "fastsem":
            logits, _, diagnostics = scheduled_logits(
                self.model, self.model.model.layers, self.model.model.config.layer_types,
                ids, mask, weights, self.spec)
        else:
            output = self.model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            logits = F.linear(output.last_hidden_state[:, -1, :], weights).squeeze(0).float()
            diagnostics = {"input_tokens": ids.shape[1], "executed_layers": 32, "compression": []}
        probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
        index = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {"prediction": labels[index], "probabilities": dict(zip(labels, probabilities)),
                "diagnostics": diagnostics}

    @staticmethod
    def _check_length(ids):
        if ids.shape[1] > 32768:
            raise ValueError("Input exceeds the tested 32768-token limit; no silent truncation")
