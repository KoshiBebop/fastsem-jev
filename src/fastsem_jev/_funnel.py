"""Frozen cosine token funnel extracted from the evaluated research implementation."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from ._core import DIRECT_SYSTEM, LETTERS, direct_messages
from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask, create_recurrent_attention_mask
GLOBAL_PREFIX_ANCHORS = 4
SPAN_RADIUS = 2

def encode_with_evidence_mask(
    tokenizer, row: dict, prompt_style: str = "direct"
) -> tuple[torch.Tensor, list[int], torch.Tensor, torch.Tensor]:
    """Encode exactly the regular prompt and identify only its evidence value tokens."""
    if prompt_style in {"compact", "compact_full_system"}:
        state = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False)
        options = "\n".join(
            f"{LETTERS[index]}: {option['description']}" for index, option in enumerate(row["options"])
        )
        content = f"[EVIDENCE]\n{state}\n[CRITERION]\n{row['question']}\n[OPTIONS]\n{options}"
        messages = [
            {
                "role": "system",
                "content": (
                    DIRECT_SYSTEM
                    if prompt_style == "compact_full_system"
                    else "Choose the correct listed option. Reply with its letter only."
                ),
            },
            {"role": "user", "content": content},
        ]
        marker = "[EVIDENCE]\n"
        terminator = "\n[CRITERION]"
        criterion_marker = "[CRITERION]"
    else:
        messages = direct_messages(row)
        marker = '"evidence": '
        terminator = ', "criterion":'
        criterion_marker = '"criterion":'
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    evidence_start = prompt.index(marker) + len(marker)
    evidence_end = prompt.index(terminator, evidence_start)
    criterion_start = prompt.index(criterion_marker, evidence_end)
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    prunable = [
        start < evidence_end and end > evidence_start and end > start
        for start, end in offsets
    ]
    pre_criterion_prunable = [
        index >= GLOBAL_PREFIX_ANCHORS and start < criterion_start and end > start
        for index, (start, end) in enumerate(offsets)
    ]
    if not any(prunable):
        raise ValueError("Could not map the serialized evidence to tokens")

    slots = []
    for letter in LETTERS[: len(row["options"])]:
        token_ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(token_ids) != 1 or tokenizer.decode(token_ids) != letter:
            raise ValueError(f"Candidate {letter!r} is not one round-trip token")
        slots.append(token_ids[0])
    return (
        torch.tensor([ids], dtype=torch.long),
        slots,
        torch.tensor(prunable, dtype=torch.bool),
        torch.tensor(pre_criterion_prunable, dtype=torch.bool),
    )

def select_positions(
    hidden_states: torch.Tensor,
    prunable_mask: torch.Tensor,
    retain_ratio: float,
    span_radius: int = SPAN_RADIUS,
    token_scores: torch.Tensor | None = None,
    evidence_anchor_count: int = 0,
    coverage_bins: int = 0,
    retain_count: int | None = None,
) -> torch.Tensor:
    """Select evidence spans by cosine similarity to the final decision token."""
    if not 0 < retain_ratio <= 1:
        raise ValueError("retain_ratio must be in (0, 1]")
    if evidence_anchor_count < 0:
        raise ValueError("evidence_anchor_count must be nonnegative")
    if coverage_bins < 0:
        raise ValueError("coverage_bins must be nonnegative")
    if retain_count is not None and retain_count < 1:
        raise ValueError("retain_count must be positive when specified")
    if hidden_states.ndim != 2 or hidden_states.shape[0] != prunable_mask.numel():
        raise ValueError("hidden_states and prunable_mask must have the same sequence length")
    device = hidden_states.device
    evidence = torch.nonzero(prunable_mask.to(device), as_tuple=False).flatten()
    if (retain_ratio == 1 and retain_count is None) or evidence.numel() == 0:
        return torch.arange(hidden_states.shape[0], device=device)

    target = min(
        evidence.numel(),
        retain_count if retain_count is not None else max(1, math.ceil(evidence.numel() * retain_ratio)),
    )
    if token_scores is None:
        vectors = F.normalize(hidden_states[evidence].float(), dim=-1)
        query = F.normalize(hidden_states[-1].float(), dim=-1)
        evidence_scores = vectors @ query
    else:
        if token_scores.ndim != 1 or token_scores.numel() != hidden_states.shape[0]:
            raise ValueError("token_scores must contain one score per sequence position")
        evidence_scores = token_scores.to(device)[evidence].float()
    ranked = evidence[torch.argsort(evidence_scores, descending=True)].tolist()
    evidence_list = evidence.tolist()
    evidence_set = set(evidence_list)
    anchor_each = min(evidence_anchor_count, target // 2)
    retained_evidence: set[int] = set(evidence[:anchor_each].tolist())
    retained_evidence.update(evidence[-anchor_each:].tolist() if anchor_each else [])

    # Reserve one local span per original-position bin before global ranking.
    # Bins are defined without labels and spend the same total evidence budget.
    if coverage_bins:
        scores_by_position = dict(zip(evidence_list, evidence_scores.tolist()))
        bin_count = min(coverage_bins, len(evidence_list))
        for bin_index in range(bin_count):
            start = bin_index * len(evidence_list) // bin_count
            stop = (bin_index + 1) * len(evidence_list) // bin_count
            group = evidence_list[start:stop]
            available = [position for position in group if position not in retained_evidence]
            if not available:
                continue
            center = max(available, key=scores_by_position.__getitem__)
            group_set = set(group)
            neighborhood = {
                position for position in range(center - span_radius, center + span_radius + 1)
                if position in group_set
            }
            additions = neighborhood - retained_evidence
            if len(retained_evidence) + len(additions) <= target:
                retained_evidence.update(additions)

    # Prefer contiguous semantic spans while respecting the declared token budget.
    for center in ranked:
        neighborhood = {
            position
            for position in range(center - span_radius, center + span_radius + 1)
            if position in evidence_set
        }
        additions = neighborhood - retained_evidence
        if additions and len(retained_evidence) + len(additions) <= target:
            retained_evidence.update(additions)
        if len(retained_evidence) == target:
            break
    if len(retained_evidence) < target:
        for position in ranked:
            retained_evidence.add(position)
            if len(retained_evidence) == target:
                break

    mandatory = torch.nonzero(~prunable_mask.to(device), as_tuple=False).flatten().tolist()
    return torch.tensor(sorted(set(mandatory) | retained_evidence), device=device, dtype=torch.long)

def segment_inputs(model, hidden_states: torch.Tensor, positions: torch.Tensor):
    """Build the exact native Qwen3.5 position embeddings and masks for one segment."""
    batch, length, _ = hidden_states.shape
    text_positions = positions.unsqueeze(0).expand(batch, -1)
    rope_positions = text_positions.unsqueeze(0).expand(3, -1, -1)
    position_embeddings = model.model.rotary_emb(hidden_states, rope_positions)
    attention_mask = torch.ones((batch, length), dtype=torch.long, device=hidden_states.device)
    mask_kwargs = {
        "config": model.model.config,
        "inputs_embeds": hidden_states,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "position_ids": text_positions,
    }
    masks = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
    }
    return text_positions, position_embeddings, masks

def config(key, schedule, selector='cosine', coverage=0, skip=()):
    return dict(key=key, schedule=[list(s) for s in schedule], selector=selector,
                coverage_bins=coverage, skip_layers_zero_based=list(skip),
                anchor_each=64, span_radius=2, final_depth=32,
                budget='ceil(original evidence tokens * ratio)', scope='evidence',
                readout='final_only_option_logits', merge='none')


CANDIDATES = [
    config('d16_r20', [(16, .20)]),
    config('d16_r125', [(16, .125)]),
    config('d12_r25', [(12, .25)]),
    config('d8_r25', [(8, .25)]),
    config('p8_r50_16_r25', [(8, .50), (16, .25)]),
    config('p8_r50_12_r25_20_r125', [(8, .50), (12, .25), (20, .125)]),
    config('d12_r20', [(12, .20)]),
    config('d12_r20_coverage8', [(12, .20)], coverage=8),
    config('d12_r20_multiquery', [(12, .20)], selector='multiquery'),
    config('d12_r20_delta_cos', [(12, .20)], selector='delta_cos'),
    config('d16_r25_skip21to24', [(16, .25)], skip=range(20, 24)),
    config('p8_r50_16_r25_skip21to24', [(8, .50), (16, .25)], skip=range(20, 24)),
]

def validate(spec, depth):
    schedule = spec['schedule']
    checkpoints = [s[0] for s in schedule]
    ratios = [s[1] for s in schedule]
    skipped = spec['skip_layers_zero_based']
    if (depth != 32 or spec['final_depth'] != depth or not schedule
            or checkpoints != sorted(set(checkpoints))
            or not all(0 < d < depth for d in checkpoints)
            or not all(0 < r <= 1 for r in ratios)
            or ratios != sorted(ratios, reverse=True)
            or len(set(skipped)) != len(skipped)
            or any(i < 0 or i >= depth - 8 for i in skipped)
            or any(d - 1 in skipped for d in checkpoints)
            or spec['selector'] not in {'cosine', 'multiquery', 'delta_cos'}):
        raise ValueError('Invalid frozen schedule or unsupported model depth')

def heuristic_scores(hidden, pre_hidden, mask, selector):
    """Fixed scores; ground truth, option correctness and dataset group unavailable."""
    if selector == 'cosine':
        return None
    vectors = F.normalize(hidden[0].float(), dim=-1)
    decision = vectors[-1]
    cosine = vectors @ decision
    if selector == 'multiquery':
        # Additional query from the last 16 protected suffix states, not evidence.
        evidence = torch.nonzero(mask, as_tuple=False).flatten()
        suffix = torch.arange(hidden.shape[1], device=hidden.device) > evidence[-1]
        query_positions = torch.nonzero(suffix & ~mask, as_tuple=False).flatten()[-16:]
        query = F.normalize(hidden[0, query_positions].float().mean(0), dim=0)
        return torch.maximum(cosine, vectors @ query)
    if selector == 'delta_cos':
        evidence = torch.nonzero(mask, as_tuple=False).flatten()
        delta = torch.linalg.vector_norm((hidden[0] - pre_hidden[0]).float(), dim=-1)
        # Equal-weight rank fusion avoids mixing arbitrary residual/cosine scales.
        # Rank only evidence; ascending rank means larger values get higher priority.
        c_rank = torch.argsort(torch.argsort(cosine[evidence], stable=True), stable=True)
        d_rank = torch.argsort(torch.argsort(delta[evidence], stable=True), stable=True)
        scores = torch.zeros_like(cosine)
        scores[evidence] = (c_rank.float() + d_rank.float()) * .5
        return scores
    raise ValueError(selector)

def scheduled_logits(model, layers, layer_types, input_ids, evidence_mask, weights, spec):
    """One forward path; each retained position remains its ORIGINAL RoPE index."""
    validate(spec, len(layers))
    hidden = model.model.embed_tokens(input_ids)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    active_mask = evidence_mask.to(input_ids.device)
    original_evidence = int(active_mask.sum())
    schedule = dict(spec['schedule'])
    skipped = set(spec['skip_layers_zero_based'])
    text_positions, rope, masks = segment_inputs(model, hidden, positions)
    trace = []
    token_layers = 0
    for index, layer in enumerate(layers):
        if index in skipped:
            continue
        token_layers += hidden.shape[1]
        pre_hidden = hidden if index + 1 in schedule else None
        hidden = layer(hidden, position_embeddings=rope,
                       attention_mask=masks[layer_types[index]], position_ids=text_positions,
                       past_key_values=None, use_cache=False)
        if index + 1 not in schedule:
            continue
        ratio = schedule[index + 1]
        target = max(1, math.ceil(original_evidence * ratio))
        scores = heuristic_scores(hidden, pre_hidden, active_mask, spec['selector'])
        chosen = select_positions(hidden[0], active_mask, ratio,
                                  retain_count=target, token_scores=scores,
                                  evidence_anchor_count=spec['anchor_each'],
                                  coverage_bins=spec['coverage_bins'], span_radius=spec['span_radius'])
        hidden = hidden.index_select(1, chosen)
        positions = positions.index_select(0, chosen)
        active_mask = active_mask.index_select(0, chosen)
        trace.append({'after_layer': index + 1, 'total_tokens': positions.numel(),
                      'evidence_tokens': min(target, original_evidence)})
        text_positions, rope, masks = segment_inputs(model, hidden, positions)
    normalized = model.model.norm(hidden)
    logits = F.linear(normalized[:, -1, :], weights).squeeze(0).float()
    diagnostics = dict(input_tokens=input_ids.shape[1], original_evidence_tokens=original_evidence,
                       compression=trace, executed_layers=len(layers) - len(skipped),
                       token_layer_fraction=token_layers / (input_ids.shape[1] * len(layers)))
    return logits, positions, diagnostics
