#!/usr/bin/env python3
"""Train one locked self-distillation arm from scored EESD correction trajectories.

The supervised pull is scaled by the precomputed trajectory weight. Selected EESD
arms additionally use forward KL from the frozen previous policy to the student on
response tokens. The explicit eed_no_anchor ablation sets this KL coefficient to zero.

This runner trains LoRA parameters only and keeps the quantized base model frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml


ANCHORED_RULES = {
    "fixed_mass_dirichlet",
    "eed_mean_no_uncertainty",
    "eesd_full",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


OPTIMIZER_SPEC = {
    "name": "torch.optim.AdamW", "betas": [0.9, 0.999], "eps": 1e-8,
    "weight_decay": 0.0, "max_grad_norm": 1.0, "lr_scheduler": "constant",
}
SCHEDULE_SCHEMA = "eesd-scheduled-response-tokens-v1"


def lora_parameters(config: dict) -> dict:
    """Report the loaded adapter's configuration, including non-default variants."""
    defaults = {"lora_dropout": 0.0, "bias": "none", "use_rslora": False,
                "use_dora": False, "rank_pattern": {}, "alpha_pattern": {},
                "modules_to_save": None, "fan_in_fan_out": False}
    result = {key: config.get(key, default) for key, default in defaults.items()}
    result.update({key: config.get(key) for key in
                   ("r", "lora_alpha", "target_modules", "peft_type", "task_type")})
    if isinstance(result["target_modules"], (set, list, tuple)):
        result["target_modules"] = sorted(result["target_modules"])
    return result


def schedule_step_bytes(row: dict, encoded: dict, token_count: int) -> bytes:
    """Bind actual order and token prefixes without arm-specific CE weights."""
    if type(token_count) is not int or not 0 < token_count <= len(encoded["completion_ids"]):
        raise ValueError("invalid scheduled response prefix")
    step = {"trajectory_id": row["trajectory_id"],
            "source_component_id": row["source_component_id"],
            "prompt_token_ids": encoded["prompt_ids"],
            "response_token_ids": encoded["completion_ids"][:token_count]}
    return (json.dumps(step, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def load_rows(path: Path, rule: str):
    from eesd.training_outcomes import read_training_rows

    return read_training_rows(path, rule)[0]


def scheduled_training_indices(rows, *, include_zero_weight_kl=False):
    """Rebuttal: all arms see identical correction identities, even if CE weight is zero."""
    return [
        i for i, row in enumerate(rows)
        if row["split"] == "train"
        and (include_zero_weight_kl or row["_weight"] > 0)
    ]


def training_schedule(
    lengths, *, seed, max_steps, gradient_accumulation, response_token_budget=None
):
    """Yield (row index, usable response tokens), cutting only the final response."""
    if not lengths or any(n < 1 for n in lengths):
        raise ValueError("positive response lengths are required")
    if max_steps < 1 or gradient_accumulation < 1:
        raise ValueError("positive step and accumulation limits required")
    if response_token_budget is not None and response_token_budget < 1:
        raise ValueError("response token budget must be positive")
    rng = random.Random(seed)
    order = list(range(len(lengths)))
    cursor = len(order)
    consumed = 0
    micro_steps = 0
    while (
        consumed < response_token_budget
        if response_token_budget is not None
        else micro_steps < max_steps * gradient_accumulation
    ):
        if cursor == len(order):
            rng.shuffle(order)
            cursor = 0
        index = order[cursor]
        cursor += 1
        count = lengths[index]
        if response_token_budget is not None:
            count = min(count, response_token_budget - consumed)
        yield index, count
        consumed += count
        micro_steps += 1


def normalize_gradients(parameters, *, denominator):
    if denominator <= 0:
        raise ValueError("gradient denominator must be positive")
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(denominator)


def encode_training_row(tokenizer, row, *, max_length, model_id):
    prompt, correction = row["prompt"], row["correction"]
    if (
        not isinstance(prompt, str)
        or not isinstance(correction, str)
        or not correction.strip()
    ):
        raise ValueError("prompt/correction must be nonempty strings")
    kwargs = {"enable_thinking": False} if "Qwen3" in model_id else {}
    messages = row.get("messages", [{"role": "user", "content": prompt}])
    if (
        not isinstance(messages, list)
        or not messages
        or any(
            not isinstance(m, dict)
            or m.get("role") not in {"system", "user"}
            or not isinstance(m.get("content"), str)
            for m in messages
        )
        or messages[-1] != {"role": "user", "content": prompt}
    ):
        raise ValueError("generation messages do not match the saved user prompt")
    prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )
    prompt_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    if "messages" in row:
        token_sha = hashlib.sha256(
            json.dumps(prompt_ids, separators=(",", ":")).encode()
        ).hexdigest()
        if row.get("prompt_token_ids_sha256") != token_sha or row.get(
            "prompt_metadata", {}
        ).get("input_tokens") != len(prompt_ids):
            raise ValueError("generation/training prompt token identity mismatch")
    if not prompt_ids:
        raise ValueError("empty prompt token sequence")
    completion_ids = tokenizer(
        correction + (tokenizer.eos_token or ""), add_special_tokens=False
    )["input_ids"]
    room = max_length - len(prompt_ids)
    if room <= 0:
        raise ValueError("prompt exceeds max_length before response")
    if not completion_ids:
        raise ValueError("correction has no trainable tokens")
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids[:room],
        "response_tokens_before_truncation": len(completion_ids),
        "response_truncated": len(completion_ids) > room,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--rule", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--previous-adapter", type=Path)
    p.add_argument("--seed", type=int, default=1701)
    p.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Legacy stopping budget; used only without --response-token-budget",
    )
    p.add_argument(
        "--response-token-budget",
        type=int,
        help="Exact response loss-token budget; overrides max-steps stopping",
    )
    p.add_argument("--gradient-accumulation", type=int, default=16)
    p.add_argument("--max-length", type=int, default=4608)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--anchor-beta", type=float, default=0.03)
    p.add_argument(
        "--include-zero-weight-kl", action="store_true",
        help="Rebuttal exact match: schedule ALL train corrections; zero-weight CE still contributes KL."
    )
    p.add_argument(
        "--token-weighting", choices=("uniform", "edit", "random"),
        default="uniform",
        help="Original EED ablation: uniform CE, changed-token highlight, or same-size random token highlight.",
    )
    p.add_argument(
        "--token-edit-boost", type=float, default=2.0,
        help="Edit vs copied token intensity; all multipliers retain exactly unit mean per scheduled response.",
    )
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    args = p.parse_args()

    from eesd.distillation import TRAIN_RULES

    if args.rule not in TRAIN_RULES or args.rule == "no_update":
        raise ValueError(
            "train one of the seven update rules; no_update evaluates the previous policy directly"
        )
    if args.include_zero_weight_kl and args.rule not in ANCHORED_RULES:
        raise ValueError("include-zero-weight-kl requires an anchored training rule")
    if args.token_weighting != "uniform" and (
        not args.include_zero_weight_kl or args.rule not in ANCHORED_RULES
    ):
        raise ValueError("rebuttal token weights require identical anchored all-row scheduling")
    if not math.isfinite(args.token_edit_boost) or args.token_edit_boost < 0:
        raise ValueError("token-edit-boost must be finite and nonnegative")
    if (
        args.max_steps < 1
        or (args.response_token_budget is not None and args.response_token_budget < 1)
        or args.gradient_accumulation < 1
        or args.max_length < 64
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or not math.isfinite(args.anchor_beta)
        or args.anchor_beta < 0
        or args.lora_r < 1
        or not math.isfinite(args.lora_alpha)
        or args.lora_alpha <= 0
    ):
        raise ValueError("invalid training hyperparameters")

    model_cfg = yaml.safe_load(args.model_config.read_text())
    if not isinstance(model_cfg, dict):
        raise ValueError("model config must be an object")
    model_id, revision = model_cfg.get("model_id"), model_cfg.get("revision")
    if (
        not isinstance(model_id, str)
        or not isinstance(revision, str)
        or len(revision) != 40
        or any(c not in "0123456789abcdef" for c in revision)
    ):
        raise ValueError(
            "model config requires pinned model_id and 40-character revision"
        )

    from eesd.training_outcomes import (
        read_training_rows,
        zero_report,
        NON_ESTIMABLE,
        protocol_binding,
        adapter_metadata,
        gemma_text_lora_targets,
        tree_sha,
    )

    previous_metadata = adapter_metadata(args.previous_adapter, args.model_config) if args.previous_adapter else None
    previous_sha_before = tree_sha(args.previous_adapter) if args.previous_adapter else None
    rows, eligibility = read_training_rows(args.input, args.rule)
    anchor_beta = args.anchor_beta if args.rule in ANCHORED_RULES else 0.0
    protocol = protocol_binding(Path(__file__).resolve().parents[1])
    if eligibility["status"] == NON_ESTIMABLE:
        report = zero_report(
            root=Path(__file__).resolve().parents[1],
            input_path=args.input,
            model_config=args.model_config,
            trainer=Path(__file__),
            rule=args.rule,
            seed=args.seed,
            budget=args.response_token_budget,
            previous_adapter=args.previous_adapter,
            anchor_beta=anchor_beta,
            eligibility=eligibility,
        )
        args.output.mkdir(parents=True, exist_ok=False)
        with (args.output / "training-outcome.json").open("x") as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report, sort_keys=True), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=False)

    import torch
    import torch.nn.functional as F
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    is_gemma3 = model_id == "google/gemma-3-4b-it"
    model_source = model_id
    tokenizer = AutoTokenizer.from_pretrained(
        model_source, revision=revision, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # Audit all arms' common train/dev population before allocating model weights.
    encoded_rows = [
        encode_training_row(
            tokenizer, row, max_length=args.max_length, model_id=model_id
        )
        for row in rows
    ]
    positive_indices = scheduled_training_indices(
        rows, include_zero_weight_kl=args.include_zero_weight_kl
    )
    positive = [rows[i] for i in positive_indices]
    positive_encoded = [encoded_rows[i] for i in positive_indices]
    development = [row for row in rows if row["split"] == "development"]
    if args.token_weighting != "uniform":
        from eesd.token_aware import original_program, edited_completion_mask
        for row, encoded in zip(positive, positive_encoded):
            source, field = original_program(row)
            encoded["edit_mask"] = edited_completion_mask(
                tokenizer, original=source, correction=row["correction"],
                target_ids=encoded["completion_ids"]
            )
            encoded["original_source_field"] = field
            encoded["edit_eos_excluded"] = bool(
                not encoded["response_truncated"]
                and tokenizer.eos_token_id is not None
                and encoded["completion_ids"][-1] == tokenizer.eos_token_id
            )
    if not positive:
        raise ValueError("no positive-weight training rows")
    if not development:
        raise ValueError(
            "development correction rows are required but never used for gradients"
        )
    token_audit = {
        "rows": len(rows),
        "max_prompt_tokens": max(len(r["prompt_ids"]) for r in encoded_rows),
        "response_truncated_rows": sum(r["response_truncated"] for r in encoded_rows),
        "max_length": args.max_length,
        "sealed_generation_message_rows": sum("messages" in row for row in rows),
        "token_weighting": args.token_weighting,
        "token_edit_boost": args.token_edit_boost,
        "edit_mask_eligible_train_rows": sum("edit_mask" in r for r in positive_encoded),
        "nontrivial_edit_train_rows": sum(
            "edit_mask" in r and 0 < int(r["edit_mask"].sum()) < len(r["edit_mask"])
            for r in positive_encoded
        ),
        "mean_changed_token_fraction_train": (
            sum(float(r["edit_mask"].mean()) for r in positive_encoded)
            / len(positive_encoded) if args.token_weighting != "uniform" else None
        ),
    }
    (args.output / "tokenization-audit.json").write_text(
        json.dumps(token_audit, indent=2) + "\n"
    )

    if is_gemma3:
        from transformers import Gemma3ForConditionalGeneration
    model_class = Gemma3ForConditionalGeneration if is_gemma3 else AutoModelForCausalLM

    def load_base():
        return model_class.from_pretrained(
            model_source,
            revision=revision,
            local_files_only=True,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            quantization_config=quant,
        )

    student_base = prepare_model_for_kbit_training(load_base())
    student_base.gradient_checkpointing_enable()
    lora_targets = (
        gemma_text_lora_targets(name for name, _ in student_base.named_modules())
        if is_gemma3 else "all-linear"
    )
    if args.previous_adapter:
        student = PeftModel.from_pretrained(
            student_base, str(args.previous_adapter), is_trainable=True
        )
    else:
        student = get_peft_model(
            student_base,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_targets,
                revision=revision,
            ),
        )
    actual_lora = lora_parameters(student.peft_config["default"].to_dict())
    if previous_metadata is not None and actual_lora != lora_parameters(previous_metadata):
        raise ValueError("loaded adapter LoRA parameters differ from the frozen checkpoint")
    student.train()

    reference = None
    if anchor_beta > 0:
        reference_base = load_base()
        if args.previous_adapter:
            reference = PeftModel.from_pretrained(
                reference_base, str(args.previous_adapter), is_trainable=False
            )
        else:
            reference = reference_base
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)

    trainable = [
        parameter for parameter in student.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=OPTIMIZER_SPEC["weight_decay"],
        betas=tuple(OPTIMIZER_SPEC["betas"]), eps=OPTIMIZER_SPEC["eps"],
    )

    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 0
    micro_steps = 0
    response_tokens = 0
    group_tokens = 0
    group_examples = 0
    totals = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "weight": 0.0}
    token_totals = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "weight": 0.0}
    schedule = training_schedule(
        [len(r["completion_ids"]) for r in positive_encoded],
        seed=args.seed,
        max_steps=args.max_steps,
        gradient_accumulation=args.gradient_accumulation,
        response_token_budget=args.response_token_budget,
    )
    schedule_digest = hashlib.sha256((SCHEDULE_SCHEMA + "\n").encode())
    for index, token_count in schedule:
        row = positive[index]
        encoded = positive_encoded[index]
        schedule_digest.update(schedule_step_bytes(row, encoded, token_count))
        prompt_ids = encoded["prompt_ids"]
        completion_ids = encoded["completion_ids"][:token_count]
        input_ids = torch.tensor(
            [prompt_ids + completion_ids], dtype=torch.long, device=student.device
        )
        labels = torch.tensor(
            [[-100] * len(prompt_ids) + completion_ids],
            dtype=torch.long,
            device=student.device,
        )
        attention_mask = torch.ones_like(input_ids)
        output = student(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
        shift_logits = output.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels.ne(-100)
        if int(mask.sum().item()) != token_count:
            raise ValueError("response loss mask does not match scheduled token budget")
        ce_tokens = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]).float(),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).reshape_as(shift_labels)
        if args.token_weighting == "uniform":
            # Preserve the exact published whole-correction CE behavior.
            ce = ce_tokens[mask].mean()
        else:
            from eesd.token_aware import prefix_multipliers
            multipliers = prefix_multipliers(
                encoded["edit_mask"], count=token_count,
                boost=args.token_edit_boost, mode=args.token_weighting,
                identity=row["trajectory_id"], seed=args.seed,
                eos_excluded=encoded["edit_eos_excluded"],
            )
            # Mean multiplier is exactly one on THIS scheduled (possibly
            # shortened) prefix; thus only token allocation differs.
            factors = torch.as_tensor(
                multipliers, device=ce_tokens.device, dtype=ce_tokens.dtype
            )
            ce = (ce_tokens[mask] * factors).mean()

        kl = torch.zeros((), device=ce.device, dtype=torch.float32)
        if reference is not None:
            with torch.no_grad():
                ref_logits = reference(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits[:, :-1, :]
            # Compute only at response positions. Full-vocabulary forward KL keeps the
            # previous policy as the anchor without selecting favored tokens.
            s = shift_logits[mask].float()
            r = ref_logits[mask].float()
            r_log = F.log_softmax(r, dim=-1)
            s_log = F.log_softmax(s, dim=-1)
            kl = (r_log.exp() * (r_log - s_log)).sum(dim=-1).mean()

        weight = float(row["_weight"])
        loss = weight * ce + anchor_beta * kl
        # In exact-budget mode each response token contributes equally before
        # trajectory weighting. Normalize by the actual group count at the step,
        # including a partially filled final accumulation group.
        (
            loss * (token_count if args.response_token_budget is not None else 1)
        ).backward()
        response_tokens += token_count
        group_tokens += token_count
        group_examples += 1
        micro_steps += 1
        totals["loss"] += float(loss.detach())
        totals["ce"] += float(ce.detach())
        totals["kl"] += float(kl.detach())
        totals["weight"] += weight
        for name, value in (("loss", loss), ("ce", ce), ("kl", kl)):
            token_totals[name] += float(value.detach()) * token_count
        token_totals["weight"] += weight * token_count

        final_budget = (
            response_tokens == args.response_token_budget
            if args.response_token_budget is not None
            else micro_steps == args.max_steps * args.gradient_accumulation
        )
        if group_examples == args.gradient_accumulation or final_budget:
            normalize_gradients(
                trainable,
                denominator=(
                    group_tokens
                    if args.response_token_budget is not None
                    else group_examples
                ),
            )
            torch.nn.utils.clip_grad_norm_(trainable, OPTIMIZER_SPEC["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            group_tokens = group_examples = 0
            if optimizer_steps % 10 == 0 or final_budget:
                print(
                    json.dumps(
                        {
                            "optimizer_step": optimizer_steps,
                            "micro_steps": micro_steps,
                            "response_tokens": response_tokens,
                            "mean_loss": totals["loss"] / micro_steps,
                            "mean_ce": totals["ce"] / micro_steps,
                            "mean_kl": totals["kl"] / micro_steps,
                            "mean_weight": totals["weight"] / micro_steps,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    if (
        args.response_token_budget is not None
        and response_tokens != args.response_token_budget
    ):
        raise RuntimeError("response token budget mismatch")
    previous_sha_after = tree_sha(args.previous_adapter) if args.previous_adapter else None
    if previous_sha_after != previous_sha_before:
        raise RuntimeError("shared previous adapter changed during training")
    adapter_dir = args.output / "adapter"
    student.save_pretrained(adapter_dir)
    report = {
        "schema": "eesd-weighted-sft-v1",
        **protocol,
        "rule": args.rule,
        "model_id": model_id,
        "revision": revision,
        "input_sha256": sha(args.input),
        "model_config_sha256": sha(args.model_config),
        "previous_adapter": str(args.previous_adapter)
        if args.previous_adapter
        else None,
        "previous_adapter_tree_sha256_before": previous_sha_before,
        "previous_adapter_tree_sha256_after": previous_sha_after,
        "training_schedule_schema": SCHEDULE_SCHEMA,
        "training_schedule_sha256": schedule_digest.hexdigest(),
        "optimizer": {**OPTIMIZER_SPEC, "learning_rate": args.learning_rate},
        "lora_config": actual_lora,
        "dependency_versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "bitsandbytes")
        },
        "seed": args.seed,
        "positive_training_trajectories": sum(row["_weight"] > 0 for row in positive),
        "scheduled_training_trajectories": len(positive),
        "scheduled_zero_weight_trajectories": sum(row["_weight"] == 0 for row in positive),
        "include_zero_weight_kl": bool(args.include_zero_weight_kl),
        "token_weighting": args.token_weighting,
        "token_edit_boost": args.token_edit_boost,
        "development_trajectories": len(development),
        "all_trajectories": len(rows),
        "sources": len({row["source_component_id"] for row in positive}),
        "max_steps": args.max_steps,
        "response_token_budget": args.response_token_budget,
        "response_tokens": response_tokens,
        "optimizer_steps": optimizer_steps,
        "micro_steps": micro_steps,
        "loss_normalization": "response_tokens"
        if args.response_token_budget is not None
        else "trajectories",
        "budget_policy": "exact response tokens; final response prefix cut"
        if args.response_token_budget is not None
        else "legacy optimizer steps",
        "tokenization_audit": token_audit,
        "trainer_source_sha256": sha(Path(__file__)),
        "gradient_accumulation": args.gradient_accumulation,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "anchor_beta": anchor_beta,
        "lora_r": actual_lora["r"],
        "lora_alpha": actual_lora["lora_alpha"],
        "lora_dropout": actual_lora["lora_dropout"],
        "token_mean_loss": token_totals["loss"] / response_tokens,
        "token_mean_ce": token_totals["ce"] / response_tokens,
        "token_mean_kl": token_totals["kl"] / response_tokens,
        "token_mean_training_weight": token_totals["weight"] / response_tokens,
        "legacy_mean_fields_unit": "microbatch / trajectory",
        "mean_loss": totals["loss"] / micro_steps,
        "mean_ce": totals["ce"] / micro_steps,
        "mean_kl": totals["kl"] / micro_steps,
        "mean_training_weight": totals["weight"] / micro_steps,
    }
    (args.output / "training-report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
