#!/usr/bin/env python3
"""One-model, one-dataset recursive EESD / matched SDPO-style baseline.

Arms: no_update, sdpo, loo, shapley. Uses only buggy code + four public
RunBugRun tests for training. NEVER consults fixed_code or evaluation tests in
self-teacher creation. Run --dry-run before reserving H100 time.

Requires bubblewrap for all generated-program execution; fails closed otherwise.
This is a research implementation; official SDPO's EMA/JSD/IS extras are NOT
reproduced. The sdpo arm is a matched full-feedback reverse-KL baseline.
"""

from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import yaml

from eesd.recursive_credit import coalition_masks, credit_from_scores
from eesd import recursive_execution
from eesd.recursive_execution import run_test, feedback_for_tests

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_records(path: Path, keys: set[str]) -> list[dict]:
    rows = []
    seen = set()
    for line in path.read_text().split('\n'):
        if not line.strip():
            continue
        item = json.loads(line)
        if set(item) != keys:
            raise ValueError(f"wrong schema in {path}: {set(item) ^ keys}")
        if (
            not isinstance(item["source_id"], str)
            or not item["source_id"]
            or item["source_id"] in seen
        ):
            raise ValueError("duplicate/invalid source_id")
        seen.add(item["source_id"])
        rows.append(item)
    if not rows:
        raise ValueError(f"empty data: {path}")
    return rows


def read_dataset(data_dir: Path, train_limit: int, eval_limit: int):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest.get("schema") != "eesd-runbugrun-public-private-v1" or not manifest.get(
        "never_read_fixed_code"
    ):
        raise ValueError("wrong RunBugRun provenance; must exclude fixed_code")
    for name, expected in manifest["file_sha256"].items():
        if digest(data_dir / name) != expected:
            raise ValueError(f"data checksum mismatch: {name}")
    train = read_records(
        data_dir / "train_public.jsonl",
        {"source_id", "problem_id", "task", "buggy_code", "public_tests"},
    )
    test = read_records(
        data_dir / "eval_private.jsonl",
        {"source_id", "problem_id", "task", "buggy_code", "evaluation_tests"},
    )
    if {r["problem_id"] for r in train} & {r["problem_id"] for r in test}:
        raise ValueError("train/eval problem_id contamination")
    if len({r["problem_id"] for r in train}) != len(train) or len(
        {r["problem_id"] for r in test}
    ) != len(test):
        raise ValueError("duplicate problem_id within split")
    if any(
        len(r["public_tests"]) != 4 or len({t["id"] for t in r["public_tests"]}) != 4
        for r in train
    ):
        raise ValueError("expected four distinct public tests per source")
    if (
        train_limit < 1
        or eval_limit < 1
        or train_limit > len(train)
        or eval_limit > len(test)
    ):
        raise ValueError("requested task count exceeds frozen data")
    return train[:train_limit], test[:eval_limit], manifest


def prompt_messages(row: dict, feedback: list[dict] | None = None) -> list[dict]:
    task = row["task"]
    buggy = row["buggy_code"]
    if not isinstance(task, str) or not isinstance(buggy, str):
        raise ValueError("task and buggy code must be strings")
    content = (
        "Repair the following buggy Python stdin/stdout program. Return a "
        "complete corrected Python script only, without markdown. "
        "Do not add examples, prose or test harness.\n\n"
        "Problem:\n" + task + "\n\nBuggy code:\n" + buggy
    )
    if feedback is not None:
        # There is intentionally NO gold corrected source in any feedback.
        content += (
            "\n\nExecution feedback from a previous attempted repair follows. "
            "Use it to reconsider the program behavior, and write a complete "
            "Python repair:\n"
            + json.dumps(feedback, sort_keys=True, ensure_ascii=False)
        )
    return [
        {"role": "system", "content": "You are an expert Python program repair model."},
        {"role": "user", "content": content},
    ]


def prompt_ids(tokenizer, row, feedback=None):
    text = tokenizer.apply_chat_template(
        prompt_messages(row, feedback), tokenize=False, add_generation_prompt=True
    )
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def response_token_ids(
    ids: list[int], eos_token_id: int, pad_token_id: int
) -> list[int]:
    out = []
    for token in ids:
        if token == pad_token_id and token != eos_token_id:
            break
        out.append(int(token))
        if token == eos_token_id:
            break
    if not out:
        raise ValueError("empty generated response")
    return out


def _extract_python(text: str) -> str:
    """Recover clean runnable Python from an Instruct model's chat-style reply.

    The model often wraps its answer in markdown fences (```python ... ```)
    despite the prompt. Strip fences and keep the largest code block; if there
    are none, strip stray fence markers and return the raw text.
    """
    import re

    blocks = re.findall(r"```[a-zA-Z0-9_+#-]*[ \t]*\n(.*?)```", text, flags=re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    cleaned = re.sub(r"```[a-zA-Z0-9_+#-]*", "", text).strip()
    return cleaned


def _require_model_config(path: Path) -> dict:
    model = yaml.safe_load(path.read_text())
    if model.get("model_id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        raise ValueError("this prospective experiment pins exactly one model/revision")
    return model


def validate_plan(args) -> dict:
    if args.arm not in ("no_update", "sdpo", "loo", "shapley") or args.rounds != 3:
        raise ValueError("frozen pilot requires three recursive rounds and a valid arm")
    if (
        args.test_count != 4
        or args.rollouts < 2
        or args.train_limit < 1
        or args.eval_limit < 1
    ):
        raise ValueError("invalid frozen experiment dimensions")
    train, test, manifest = read_dataset(
        args.data_dir, args.train_limit, args.eval_limit
    )
    model = _require_model_config(args.model_config)
    return {
        "schema": "eesd-recursive-shapley-plan-v1",
        "status": "planned_not_executed",
        "arm": args.arm,
        "dataset": "RunBugRun Python official SQLite",
        "model_id": model["model_id"],
        "model_revision": model["revision"],
        "public_tests": 4,
        "rollouts": args.rollouts,
        "rounds": args.rounds,
        "train_sources": len(train),
        "eval_sources": len(test),
        "teacher_coalitions_per_rollout": len(coalition_masks(4, args.arm))
        if args.arm != "no_update"
        else 0,
        "teacher_subset_forward_estimate": args.rounds
        * len(train)
        * args.rollouts
        * (
            len(set(coalition_masks(4, args.arm)) | {0})
            if args.arm != "no_update"
            else 0
        ),
        "data_manifest_sha256": digest(args.data_dir / "manifest.json"),
        "initial_adapter": str(args.init_adapter)
        if args.init_adapter
        else "deterministically initialized",
        "no_gold_code": True,
        "no_eval_feedback_in_training": True,
        "baseline_definition": "sdpo-style reverse KL, frozen full-feedback teacher (not official SDPO reproduction)",
    }


def _models(args):
    import torch
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    config = _require_model_config(args.model_config)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    def base():
        value = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            device_map={"": 0},
            torch_dtype=dtype,
            quantization_config=quant,
        )
        return prepare_model_for_kbit_training(value)

    def attach(adapter: Path | None, trainable: bool):
        raw = base()
        if adapter is None:
            m = get_peft_model(
                raw,
                LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.0,
                    target_modules=[
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )
        else:
            m = PeftModel.from_pretrained(
                raw, str(adapter), is_trainable=trainable, local_files_only=True
            )
        if not trainable:
            for p in m.parameters():
                p.requires_grad_(False)
            m.eval()
        return m

    return torch, tokenizer, attach


def _logprobs_for_fixed_response(
    model, tokenizer, row, feedback, response_ids, *, return_distribution: bool
):
    import torch
    import torch.nn.functional as F

    context = prompt_ids(tokenizer, row, feedback)
    full = torch.tensor([context + response_ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        logits = (
            model(input_ids=full, use_cache=False)
            .logits[0, len(context) - 1 : -1]
            .float()
        )
        chosen = torch.as_tensor(response_ids, dtype=torch.long, device=model.device)
        selected = logits.gather(1, chosen.unsqueeze(1)).squeeze(1) - torch.logsumexp(
            logits, dim=-1
        )
        result = selected.detach().cpu().numpy()
        dist = F.log_softmax(logits, dim=-1).detach() if return_distribution else None
    return result, dist


def _teacher_for_rollout(teacher, tokenizer, row, response_ids, feedback, arm):
    tests = len(feedback)
    full = (1 << tests) - 1
    masks = set(coalition_masks(tests, arm)) | {0}
    scores = np.zeros((1 << tests, len(response_ids)), dtype=np.float64)
    baseline = None
    target = None
    for mask in sorted(masks):
        chosen = [feedback[j] for j in range(tests) if mask & (1 << j)]
        # Mask 0 is truly absent feedback (not an empty feedback message).
        val, dist = _logprobs_for_fixed_response(
            teacher,
            tokenizer,
            row,
            None if mask == 0 else chosen,
            response_ids,
            return_distribution=mask in (0, full),
        )
        scores[mask] = val
        if mask == 0:
            baseline = dist
        if mask == full:
            target = dist
    if arm == "sdpo":
        weights = np.ones(len(response_ids), dtype=np.float32)
        info = {
            "active_fraction": 1.0,
            "effective_mass_mean": None,
            "negative_shapley_fraction": None,
            "zero_feedback_signal": False,
        }
    else:
        _, weights, info = credit_from_scores(scores, tests, method=arm)
    return baseline, target, weights, info


def _distill_one(
    student, old_logits, teacher_logits, weights, ids, context_len, *, beta: float
):
    import torch
    import torch.nn.functional as F

    logits = (
        student(input_ids=ids, use_cache=False).logits[0, context_len - 1 : -1].float()
    )
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    logq = teacher_logits.detach().to(logp.device)
    logold = old_logits.detach().to(logp.device)
    # Consistent token-level REVERSE KL (student || frozen feedback teacher).
    loss_per_token = (p * (logp - logq)).sum(dim=-1)
    anchor = (p * (logp - logold)).sum(dim=-1)
    w = torch.as_tensor(weights, device=logp.device, dtype=torch.float32)
    if len(w) != len(loss_per_token):
        raise ValueError("teacher/student response misalignment")
    primary = (loss_per_token * w).sum() / w.sum().clamp(min=1.0)
    loss = primary + beta * anchor.mean()
    return loss, {
        "distillation_kl": float(primary.detach().cpu()),
        "anchor_kl": float(anchor.mean().detach().cpu()),
    }


def _sample_rollouts(teacher, tokenizer, row, args, round_number):
    import torch

    context = prompt_ids(tokenizer, row)
    if len(context) > args.max_prompt_tokens:
        raise ValueError(
            f"input prompt longer than frozen token budget: {row['source_id']}"
        )
    prompt = torch.tensor([context], dtype=torch.long, device=teacher.device)
    generator_seed = int(
        hashlib.sha256(
            f"{args.seed}:{round_number}:{row['source_id']}".encode()
        ).hexdigest()[:8],
        16,
    )
    torch.manual_seed(generator_seed)
    torch.cuda.manual_seed_all(generator_seed)
    result = []
    # Small generation batches avoid an H100-specific peak-memory requirement.
    for start in range(0, args.rollouts, args.generate_batch):
        count = min(args.generate_batch, args.rollouts - start)
        with torch.no_grad():
            generated = teacher.generate(
                prompt,
                attention_mask=torch.ones_like(prompt),
                do_sample=True,
                temperature=args.temperature,
                top_p=0.95,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=count,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for seq in generated:
            tokens = response_token_ids(
                seq[len(context) :].tolist(),
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
            )
            result.append(
                (
                    tokens,
                    _extract_python(tokenizer.decode(tokens, skip_special_tokens=True)),
                )
            )
    return result


def _evaluate(model, tokenizer, rows, args, round_number):
    import torch

    model.eval()
    results = []
    for row in rows:
        context = prompt_ids(tokenizer, row)
        if len(context) > args.max_prompt_tokens:
            raise ValueError(f"oversized evaluation prompt: {row['source_id']}")
        inp = torch.tensor([context], dtype=torch.long, device=model.device)
        with torch.no_grad():
            seq = model.generate(
                inp,
                attention_mask=torch.ones_like(inp),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )[0]
        code = _extract_python(
            tokenizer.decode(seq[len(context) :], skip_special_tokens=True)
        )
        flags = [run_test(code, t).status == "PASS" for t in row["evaluation_tests"]]
        results.append(
            {
                "source_id": row["source_id"],
                "problem_id": row["problem_id"],
                "all_pass": all(flags),
                "passed": sum(flags),
                "tests": len(flags),
                "generated_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            }
        )
    return {
        "schema": "eesd-fresh-policy-eval-v1",
        "round": round_number,
        "sources": len(results),
        "pass_at_1": sum(r["all_pass"] for r in results) / len(results),
        "mean_test_pass_fraction": sum(r["passed"] / r["tests"] for r in results)
        / len(results),
        "rows": results,
        "selection": "single greedy candidate, no feedback, unobserved evaluation tests",
    }


def _save_json(path: Path, payload: dict) -> None:
    with path.open("x") as f:
        json.dump(payload, f, ensure_ascii=False, sort_keys=True, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-config", type=Path, default=ROOT / "configs/models/qwen25_7b.yaml"
    )
    parser.add_argument(
        "--arm", choices=["no_update", "sdpo", "loo", "shapley"], required=True
    )
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=4)
    parser.add_argument("--train-limit", type=int, default=64)
    parser.add_argument("--eval-limit", type=int, default=64)
    parser.add_argument("--generate-batch", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = validate_plan(args)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if args.output.exists():
        raise FileExistsError(
            "output already exists; never overwrite experimental result"
        )
    if args.generate_batch < 1 or args.beta < 0 or args.max_new_tokens < 1:
        raise ValueError("invalid compute configuration")
    # Preflight sandbox before allocating 7B model weights; bwrap unavailable is fatal.
    smoke = run_test("print(47)", {"input": "", "expected": "47"})
    if smoke.status != "PASS":
        raise RuntimeError(f"sandbox positive control failed: {smoke}")
    plan["sandbox_mode"] = recursive_execution.SANDBOX_MODE
    plan["sandbox_note"] = recursive_execution.SANDBOX_NOTE
    import torch

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "requires BF16 CUDA hardware; no CPU imitation of model results"
        )
    train_rows, eval_rows, manifest = read_dataset(
        args.data_dir, args.train_limit, args.eval_limit
    )
    torch, tokenizer, attach = _models(args)
    args.output.mkdir(parents=True, exist_ok=False)
    _save_json(args.output / "plan.json", plan)
    if args.init_adapter:
        if not (args.init_adapter / "adapter_config.json").exists():
            raise ValueError("initial checkpoint missing adapter config")
        prev_adapter = args.init_adapter
    else:
        init = attach(None, trainable=False)
        prev_adapter = args.output / "initial_adapter"
        init.save_pretrained(prev_adapter)
        del init
        gc.collect()
        torch.cuda.empty_cache()
    for round_idx in range(args.rounds + 1):
        rd = args.output / f"round_{round_idx}"
        rd.mkdir(exist_ok=False)
        teacher = attach(prev_adapter, trainable=False)
        ev = _evaluate(teacher, tokenizer, eval_rows, args, round_idx)
        _save_json(rd / "evaluation.json", ev)
        print(
            json.dumps(
                {
                    "round": round_idx,
                    "eval_pass_at_1": ev["pass_at_1"],
                    "eval_mean_pass": ev["mean_test_pass_fraction"],
                }
            ),
            flush=True,
        )
        if round_idx == args.rounds or args.arm == "no_update":
            del teacher
            gc.collect()
            torch.cuda.empty_cache()
            if args.arm == "no_update":
                # Same frozen model evaluated each round; no optimizer updates.
                continue
            break
        student = attach(prev_adapter, trainable=True)
        trainable = [p for p in student.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("no trainable LoRA parameters")
        optim = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
        student.train()
        records = []
        log_path = rd / "training.jsonl"
        with log_path.open("x") as stream:
            for row in train_rows:
                sampled = _sample_rollouts(teacher, tokenizer, row, args, round_idx)
                per_candidate = []
                optimizer_kl = []
                optim.zero_grad(set_to_none=True)
                for response_ids, code in sampled:
                    feedback = feedback_for_tests(code, row["public_tests"])
                    base_dist, full_dist, weights, info = _teacher_for_rollout(
                        teacher, tokenizer, row, response_ids, feedback, args.arm
                    )
                    context = prompt_ids(tokenizer, row)
                    full = torch.tensor(
                        [context + response_ids],
                        dtype=torch.long,
                        device=student.device,
                    )
                    loss, stats = _distill_one(
                        student,
                        base_dist,
                        full_dist,
                        weights,
                        full,
                        len(context),
                        beta=args.beta,
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError("nonfinite student loss")
                    (loss / args.rollouts).backward()
                    optimizer_kl.append(stats)
                    per_candidate.append(
                        {
                            "public_pass": sum(
                                f["outcome"] == "PASS" for f in feedback
                            ),
                            "outcomes": [f["outcome"] for f in feedback],
                            "candidate_sha256": hashlib.sha256(
                                code.encode()
                            ).hexdigest(),
                            "tokens": len(response_ids),
                            "credit": info,
                        }
                    )
                    del base_dist, full_dist, full, loss
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if not torch.isfinite(norm):
                    raise RuntimeError("nonfinite gradient norm")
                optim.step()
                record = {
                    "source_id": row["source_id"],
                    "round": round_idx,
                    "arm": args.arm,
                    "gradient_norm": float(norm.detach().cpu()),
                    "mean_distillation_kl": float(
                        np.mean([s["distillation_kl"] for s in optimizer_kl])
                    ),
                    "mean_anchor_kl": float(
                        np.mean([s["anchor_kl"] for s in optimizer_kl])
                    ),
                    "candidates": per_candidate,
                }
                records.append(record)
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
        updated = rd / "adapter"
        student.save_pretrained(updated)
        _save_json(
            rd / "training_summary.json",
            {
                "sources": len(records),
                "rollouts": args.rollouts,
                "public_executions": len(records) * args.rollouts * 4,
                "teacher_scoring_policy": "frozen at round start",
                "student_optimizer_steps": len(records),
                "train_public_pass_fraction": float(
                    np.mean(
                        [r["public_pass"] / 4 for x in records for r in x["candidates"]]
                    )
                ),
                "unique_code_fraction": float(
                    np.mean(
                        [
                            len({c["candidate_sha256"] for c in x["candidates"]})
                            / args.rollouts
                            for x in records
                        ]
                    )
                ),
                "token_credit_zero_fraction": float(
                    np.mean(
                        [
                            c["credit"]["zero_feedback_signal"]
                            for x in records
                            for c in x["candidates"]
                        ]
                    )
                )
                if args.arm != "sdpo"
                else None,
            },
        )
        del student, teacher, optim
        gc.collect()
        torch.cuda.empty_cache()
        prev_adapter = updated
    print(
        json.dumps(
            {"status": "completed", "arm": args.arm, "output": str(args.output)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
