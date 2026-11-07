"""Reviewer-motivated edit-aware correction supervision for original EED.

The *submitted* EED only provides the correction-level coefficient w(h).
This module supplies a deterministic, budget-neutral token multiplier g_t
and does not change four-state Dirichlet evidence, tests, the KL reference,
nor the correction target. g_t is re-normalized for each scheduled prefix:
    g_t = (1 + boost * edit_t) / (1 + boost * mean(edit_t)).
Hence mean_t g_t = 1 over the tokens used in this optimizer micro-step.

This is a **diagnostic extension**, not the original submitted method.
"""
from __future__ import annotations

import hashlib
from difflib import SequenceMatcher
import json
import math
import random
from typing import Mapping, Sequence

import numpy as np

from eesd.shapley_relevance import edited_token_masks


def original_program(record: Mapping) -> tuple[str, str]:
    """Get the exact pre-correction source; NEVER substitute prompt text blindly.

    Existing correction trajectory raw records commonly have 'original'.
    Older scored records may keep only a JSON prompt with a 'program' key.
    Fail closed for pure synthesis prompts or unknown formats; without a
    real before/after source pair the edit-aware study is not identifiable.
    """
    value = None
    field = None
    for name in ("original", "original_code"):
        candidate = record.get(name)
        if candidate is not None:
            if not isinstance(candidate, str) or not candidate.strip():
                raise ValueError(f"invalid {name}: must be nonempty code string")
            if value is not None and candidate != value:
                raise ValueError("conflicting original and original_code")
            value, field = candidate, name

    prompt = record.get("prompt")
    prompt_program = None
    if isinstance(prompt, str) and "\n" in prompt:
        # Original EESD repair prompt: one instruction line + JSON payload
        payload = prompt.split("\n", 1)[1].strip()
        try:
            body = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            body = None
        if isinstance(body, dict) and "program" in body:
            prompt_program = body["program"]
            if not isinstance(prompt_program, str) or not prompt_program.strip():
                raise ValueError("invalid prompt.program")
    if value is not None and prompt_program is not None and value != prompt_program:
        raise ValueError("raw original code differs from the prompt program")
    if value is not None:
        return value, field
    if prompt_program is not None:
        return prompt_program, "prompt.program"
    raise ValueError(
        "Token-aware EED requires original code paired with correction; "
        "expected a nonempty 'original' or 'original_code' field, or an "
        "original-repair JSON prompt with a 'program' key. "
        "Do NOT label pure synthesis data as token edit supervision."
    )


def edited_completion_mask(
    tokenizer, *, original: str, correction: str, target_ids: Sequence[int]
) -> np.ndarray:
    """Mask changed TARGET token positions using EXACT training tokenization.

    Includes the same EOS token for before/after; EOS is never a changed
    program token. Target may be truncated by max_length but must be a prefix
    of the original untruncated correction+EOS tokenization.
    """
    if not isinstance(original, str) or not isinstance(correction, str):
        raise TypeError("source and correction must be strings")
    target = list(target_ids)
    if not target or any(type(t) is not int for t in target):
        raise ValueError("nonempty integer target token sequence required")
    eos = tokenizer.eos_token or ""
    after_ids = list(tokenizer(correction + eos, add_special_tokens=False)["input_ids"])
    before_ids = list(tokenizer(original + eos, add_special_tokens=False)["input_ids"])
    if len(target) > len(after_ids) or target != after_ids[: len(target)]:
        raise ValueError("edit mask target must exactly match trainer completion tokens")
    if not before_ids:
        raise ValueError("original code has empty tokens")
    _, changed = edited_token_masks(before_ids, after_ids)
    # A pure deletion has no replaced/inserted TARGET token. The first
    # surviving token after the deletion is the decision that skips the
    # removed span. Give that boundary a credit position (never EOS);
    # if deletion is at the end, use the last surviving program token.
    has_eos = bool(
        eos and after_ids and
        after_ids[-1] == getattr(tokenizer, "eos_token_id", None)
    )
    last_program_position = len(after_ids) - (2 if has_eos else 1)
    for tag, i1, i2, j1, j2 in SequenceMatcher(
        a=before_ids, b=after_ids, autojunk=False
    ).get_opcodes():
        if tag == "delete" and last_program_position >= 0:
            boundary = min(j1, last_program_position)
            changed[boundary] = True
    changed = changed[: len(target)].astype(bool)
    # EOS is a policy-termination token, not an edit.
    if has_eos and len(target) == len(after_ids):
        changed[-1] = False
    return changed


def matched_random_mask(edit_mask: Sequence[bool], *, identity: str, seed: int,
                        eos_excluded: bool = False) -> np.ndarray:
    """Match the number of edit positions without using their location."""
    mask = np.asarray(edit_mask, dtype=bool)
    if mask.ndim != 1 or len(mask)==0:
        raise ValueError("nonempty edit mask required")
    if not isinstance(identity,str) or not identity or type(seed) is not int:
        raise ValueError("valid source identity and seed required")
    # Only exclude the last position when it is actually the EOS position.
    candidates=list(range(len(mask)-(1 if eos_excluded else 0)))
    if eos_excluded and bool(mask[-1]):
        raise ValueError("EOS cannot be an edit position")
    k=int(mask.sum())
    if k>len(candidates):
        raise ValueError("more edits than positions")
    digest=hashlib.sha256(f"{seed}:{identity}:token-random".encode()).digest()
    rng=random.Random(int.from_bytes(digest[:8],"big"))
    chosen=rng.sample(candidates,k)
    result=np.zeros(len(mask),dtype=bool)
    result[chosen]=True
    return result


def token_multipliers(mask: Sequence[bool], *, boost: float=2.0) -> np.ndarray:
    """Positive edit weights with EXACT unit mean, even when every token is edited."""
    arr=np.asarray(mask,dtype=bool)
    if arr.ndim!=1 or len(arr)==0:
        raise ValueError("nonempty 1D token edit mask required")
    if not math.isfinite(float(boost)) or boost<0:
        raise ValueError("finite nonnegative boost required")
    if boost == 0:
        return np.ones(len(arr),dtype=np.float64)
    x=arr.astype(np.float64)
    result=(1. + float(boost)*x)/(1. + float(boost)*float(x.mean()))
    if not np.isfinite(result).all() or np.any(result<=0) or not np.isclose(
        result.mean(),1.,rtol=0,atol=1e-12):
        raise RuntimeError("invalid normalized token multiplier")
    return result


def prefix_multipliers(edit_mask: Sequence[bool], *, count:int, boost:float,
                       mode:str, identity:str, seed:int,
                       eos_excluded:bool=False) -> np.ndarray:
    """Match a prefix-cropped exact response-token budget.

    The same (count, identity, seed) yields deterministic random controls.
    """
    arr=np.asarray(edit_mask,dtype=bool)
    if type(count) is not int or not 1<=count<=len(arr):
        raise ValueError("invalid scheduled completion prefix")
    if mode=="uniform":
        return np.ones(count,dtype=np.float64)
    if mode=="edit":
        selected=arr[:count]
    elif mode=="random":
        selected=matched_random_mask(
            arr[:count],identity=identity,seed=seed,
            eos_excluded=eos_excluded and count==len(arr)
        )
    else:raise ValueError(f"unknown token weighting mode {mode!r}")
    return token_multipliers(selected,boost=boost)
