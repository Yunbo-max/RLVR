#!/usr/bin/env python3
"""Revised EESD: behavior-aligned Shapley selects evidence BEFORE self-distillation.

Original negative result remains untouched. One Qwen2.5-Coder-7B-Instruct,
RunBugRun Python, G=8, four public feedback tests, four independent *training*
audit probes, three genuinely recursive rounds. No reference fixed programs.

Arms: no_update, sdpo (matched full-feedback), audit_gate (same audit budget,
no Shapley), shapley_select (exact signed test Shapley + feedback selection).
Fail closed unless proper bubblewrap isolation is available.
"""

from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_recursive_shapley_sdpo as legacy  # reuse pinned Qwen loading, LoRA and prompt templates
from eesd.behavioral_shapley import (
    select_feedback,
    full_feedback_audit_gate,
    directional_token_weights,
)
from eesd.strict_execution import run_test, feedback_for_tests, audit_vector

ARMS = ("no_update", "sdpo", "audit_gate", "shapley_select")


def load_matched_data(data_dir: Path, train_limit: int, eval_limit: int):
    train, eval_rows, manifest = legacy.read_dataset(data_dir, train_limit, eval_limit)
    if manifest.get("behavioral_schema") != "eesd-runbugrun-behavioral-v1":
        raise ValueError(
            "requires new 4+4+4 behavioral data split; old audit is not trainable"
        )
    calibration = legacy.read_records(
        data_dir / "train_calibration.jsonl",
        {"source_id", "problem_id", "calibration_tests"},
    )
    reserved = legacy.read_records(
        data_dir / "train_audit_do_not_train.jsonl",
        {"source_id", "problem_id", "audit_tests"},
    )
    by_id = {r["source_id"]: r for r in calibration}
    reserved_by_id = {r["source_id"]: r for r in reserved}
    if len(by_id) != len(calibration) or len(by_id) < len(train):
        raise ValueError("missing/duplicate training audit records")
    for row in train:
        probe = by_id.get(row["source_id"])
        heldout = reserved_by_id.get(row["source_id"])
        if (
            probe is None
            or heldout is None
            or probe["problem_id"] != row["problem_id"]
            or heldout["problem_id"] != row["problem_id"]
        ):
            raise ValueError("calibration/held-out source identity mismatch")
        if len(probe["calibration_tests"]) != 4 or len(heldout["audit_tests"]) != 4:
            raise ValueError("expected 4 calibration and 4 held-out tests")
        pub = {x["id"] for x in row["public_tests"]}
        cal = {x["id"] for x in probe["calibration_tests"]}
        aud = {x["id"] for x in heldout["audit_tests"]}
        if len(cal) != 4 or len(aud) != 4 or pub & cal or pub & aud or cal & aud:
            raise ValueError("feedback/calibration/audit test contamination")
    return train, eval_rows, by_id, manifest


def _normalized_ids(tokenizer, code: str, max_len: int) -> list[int]:
    """Distill ONLY runnable code, never model-produced markdown fences."""
    if not isinstance(code, str) or not code.strip():
        return []
    text = code.strip() + "\n"
    ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    return ids if 0 < len(ids) <= max_len else []


def _coalition_score(
    teacher, tokenizer, row, tokens, feedback, mask, max_prompt_tokens
):
    chosen = [feedback[j] for j in range(len(feedback)) if mask & (1 << j)]
    ctx = None if mask == 0 else chosen
    if len(legacy.prompt_ids(tokenizer, row, ctx)) > max_prompt_tokens:
        raise ValueError("feedback prompt exceeds fixed model input budget")
    return legacy._logprobs_for_fixed_response(
        teacher, tokenizer, row, ctx, tokens, return_distribution=False
    )[0]


def score_candidate(
    teacher, tokenizer, row, tokens, feedback, *, arm, max_prompt_tokens
) -> np.ndarray:
    ntests = len(feedback)
    if ntests != 4:
        raise ValueError("four public feedback tests required")
    result = np.zeros((1 << ntests, len(tokens)), dtype=np.float64)
    if arm == "shapley_select":
        masks = range(1 << ntests)
    else:
        masks = (0, (1 << ntests) - 1)
    for mask in masks:
        result[mask] = _coalition_score(
            teacher, tokenizer, row, tokens, feedback, mask, max_prompt_tokens
        )
    return result


def masked_teacher(teacher, tokenizer, row, tokens, feedback, mask):
    selected = [feedback[j] for j in range(4) if mask & (1 << j)]
    _, teacher_logq = legacy._logprobs_for_fixed_response(
        teacher,
        tokenizer,
        row,
        selected if mask else None,
        tokens,
        return_distribution=True,
    )
    _, old_logq = legacy._logprobs_for_fixed_response(
        teacher, tokenizer, row, None, tokens, return_distribution=True
    )
    return teacher_logq, old_logq


def distill_token_kl(
    student, old_logq, teacher_logq, weights, ids, prompt_length, *, beta
):
    """Confidence-preserving KL: never renormalize weak evidence to full strength."""
    import torch
    import torch.nn.functional as F

    if beta < 0:
        raise ValueError("negative KL anchor")
    logits = (
        student(input_ids=ids, use_cache=False)
        .logits[0, prompt_length - 1 : -1]
        .float()
    )
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    q = teacher_logq.detach().to(logp.device)
    old = old_logq.detach().to(logp.device)
    w = torch.as_tensor(weights, device=logp.device, dtype=torch.float32)
    if logits.shape != q.shape or q.shape != old.shape or len(w) != len(logp):
        raise ValueError("student/teacher token-vocab alignment mismatch")
    if not torch.isfinite(w).all() or (w < 0).any():
        raise ValueError("invalid confidence weights")
    token_kl = (p * (logp - q)).sum(-1)
    anchor = (p * (logp - old)).sum(-1)
    # Unlike earlier code, this loss is not divided by w.sum().
    loss = (w * (token_kl + beta * anchor)).mean()
    return loss, {
        "mean_weight": float(w.mean().detach().cpu()),
        "weighted_teacher_kl": float((w * token_kl).mean().detach().cpu()),
        "weighted_anchor_kl": float((w * anchor).mean().detach().cpu()),
    }


def _evaluate(model, tokenizer, rows, args, round_idx: int) -> dict:
    import torch

    model.eval()
    out = []
    for row in rows:
        ctx = legacy.prompt_ids(tokenizer, row)
        if len(ctx) > args.max_prompt_tokens:
            raise ValueError("evaluation prompt over model budget")
        inp = torch.tensor([ctx], dtype=torch.long, device=model.device)
        with torch.no_grad():
            generated = model.generate(
                inp,
                attention_mask=torch.ones_like(inp),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )[0]
        raw = tokenizer.decode(generated[len(ctx) :], skip_special_tokens=True)
        code = legacy._extract_python(raw)
        correctness = audit_vector(code, row["evaluation_tests"])
        out.append(
            {
                "source_id": row["source_id"],
                "problem_id": row["problem_id"],
                "all_pass": bool(all(correctness)),
                "passes": sum(correctness),
                "total_tests": len(correctness),
                "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            }
        )
    return {
        "schema": "eesd-behavioral-fresh-eval-v1",
        "round": round_idx,
        "sources": len(out),
        "pass_at_1": sum(r["all_pass"] for r in out) / len(out),
        "mean_test_pass_fraction": sum(r["passes"] / r["total_tests"] for r in out)
        / len(out),
        "rows": out,
    }


def _train_source(
    teacher, student, optimizer, tokenizer, row, audit_tests, args, round_idx
):
    import torch

    sampled = legacy._sample_rollouts(teacher, tokenizer, row, args, round_idx)
    clean = []
    for _, code in sampled:
        tok = _normalized_ids(tokenizer, code, args.max_new_tokens)
        if tok:
            clean.append((tok, code))
    if len(clean) < 2:
        return {
            "source_id": row["source_id"],
            "skip": "too_few_valid_programs",
            "step": False,
        }
    feedback = [feedback_for_tests(code, row["public_tests"]) for _, code in clean]
    audit = np.asarray(
        [audit_vector(code, audit_tests) for _, code in clean], dtype=np.float64
    )
    # In SDPO, audit is measured for matched execution cost only; it does NOT
    # affect the loss/target. Audit-only and Shapley arms use it for gating.
    scores = [
        score_candidate(
            teacher,
            tokenizer,
            row,
            tok,
            fb,
            arm=args.arm,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        for (tok, _), fb in zip(clean, feedback)
    ]
    # Store model-independent diagnostics LOCALLY for post-training audit.
    # Reserved audit labels never influence these scores or the optimizer.
    candidate_diagnostics = {
        "candidate_sources": [code for _, code in clean],
        "candidate_shift_means": [
            (v.mean(axis=1) - v[0].mean()).tolist() for v in scores
        ],
    }
    if args.arm == "sdpo":
        mask = 15
        weights = [np.ones(len(tok), dtype=np.float32) for tok, _ in clean]
        selection = {"reason": "all_feedback_unweighted", "phi": None, "values": None}
    else:
        if args.arm == "audit_gate":
            result = full_feedback_audit_gate(scores, audit, min_gain=args.min_gain)
        elif args.arm == "shapley_select":
            result = select_feedback(
                scores,
                audit,
                uncertainty_penalty=args.audit_uncertainty,
                subset_cost=args.subset_cost,
                min_gain=args.min_gain,
            )
        else:
            raise ValueError("invalid training arm")
        mask = result.mask
        selection = {
            "reason": result.reason,
            "phi": result.phi.tolist(),
            "values": result.value_by_subset.tolist(),
            "subset_score": result.score_by_subset.tolist(),
            "audit_contrast": result.audit_contrast,
        }
        weights = directional_token_weights(
            scores, audit, mask, shift_scale=args.shift_scale
        )
    if mask == 0 or sum(float(w.sum()) for w in weights) <= 1e-8:
        return {
            "source_id": row["source_id"],
            "step": False,
            "skip": selection["reason"],
            "selected_mask": mask,
            "audit_variance": float(audit.var(axis=0).mean()),
            "selection": selection,
            **candidate_diagnostics,
        }
    optimizer.zero_grad(set_to_none=True)
    stats = []
    applied = 0
    for (tokens, code), fb, w in zip(clean, feedback, weights):
        if not np.any(w > 0):
            continue
        logq, oldq = masked_teacher(teacher, tokenizer, row, tokens, fb, mask)
        ctx = legacy.prompt_ids(tokenizer, row)
        ids = torch.tensor([ctx + tokens], dtype=torch.long, device=student.device)
        loss, stat = distill_token_kl(
            student, oldq, logq, w, ids, len(ctx), beta=args.beta
        )
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite behavioral distillation loss")
        (loss / len(clean)).backward()
        applied += 1
        stats.append(stat)
        del ids, loss, logq, oldq
    if applied == 0:
        return {
            "source_id": row["source_id"],
            "step": False,
            "skip": "all_token_weights_zero",
        }
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in student.parameters() if p.requires_grad], 1.0
    )
    if not torch.isfinite(norm):
        raise RuntimeError("nonfinite gradient norm")
    optimizer.step()
    return {
        "source_id": row["source_id"],
        "step": True,
        "selected_mask": mask,
        "selection": selection,
        "eligible_programs": len(clean),
        "audit_pass_fraction": float(audit.mean()),
        "public_pass_fraction": sum(
            sum(f["outcome"] == "PASS" for f in fb) for fb in feedback
        )
        / (4 * len(feedback)),
        "audit_variance": float(audit.var(axis=0).mean()),
        "mean_weight": float(np.mean([s["mean_weight"] for s in stats])),
        "grad_norm": float(norm.detach().cpu()),
        "program_code_sha256": [
            hashlib.sha256(code.encode()).hexdigest() for _, code in clean
        ],
        "candidate_audit_scores": audit.mean(axis=1).tolist(),
        **candidate_diagnostics,
    }


def _save_json(path: Path, data: dict):
    with path.open("x") as output:
        json.dump(data, output, sort_keys=True, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument(
        "--model-config", type=Path, default=ROOT / "configs/models/qwen25_7b.yaml"
    )
    ap.add_argument("--init-adapter", type=Path)
    ap.add_argument("--seed", type=int, default=1701)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--train-limit", type=int, default=64)
    ap.add_argument("--eval-limit", type=int, default=64)
    ap.add_argument("--generate-batch", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-prompt-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--audit-uncertainty", type=float, default=0.5)
    ap.add_argument("--subset-cost", type=float, default=0.001)
    ap.add_argument("--min-gain", type=float, default=0.001)
    ap.add_argument("--shift-scale", type=float, default=0.15)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    legacy._require_model_config(args.model_config)
    train, ev, audit, manifest = load_matched_data(
        args.data_dir, args.train_limit, args.eval_limit
    )
    if (
        args.rounds != 3
        or args.rollouts < 2
        or args.generate_batch < 1
        or args.beta < 0
    ):
        raise ValueError("frozen protocol requires 3 rounds and >=2 rollouts")
    plan = {
        "schema": "eesd-behavioral-shapley-v1",
        "status": "preflight_not_run",
        "arm": args.arm,
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "revision": legacy.MODEL_REVISION,
        "train_sources": len(train),
        "eval_sources": len(ev),
        "dataset_manifest_sha256": legacy.digest(args.data_dir / "manifest.json"),
        "calibration_budget_per_rollout": len(
            next(iter(audit.values()))["calibration_tests"]
        ),
        "reserved_train_audit_tests_per_source": 4,
        "feedback_tests": 4,
        "rollouts": args.rollouts,
        "rounds": args.rounds,
        "isolation": "bubblewrap_mandatory_no_soft_fallback",
        "gold_fixed_code_used": False,
        "calibration_probes_are_training_side_labels": True,
        "reserved_train_audit_used_for_training": False,
        "official_sdpo_reproduction": False,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "init_adapter": str(args.init_adapter) if args.init_adapter else None,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if args.output.exists():
        raise FileExistsError("experimental result directory already exists")
    smoke = run_test("print(47)", {"input": "", "expected": "47"})
    if smoke.status != "PASS":
        raise RuntimeError("secure sandbox positive control failed")
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("must have BF16 CUDA GPU; this run will not fake results")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch, tokenizer, attach = legacy._models(args)
    args.output.mkdir(parents=True, exist_ok=False)
    plan["status"] = "executing"
    _save_json(args.output / "plan.json", plan)
    if args.init_adapter:
        prev = args.init_adapter
        if not (prev / "adapter_config.json").is_file():
            raise FileNotFoundError("initial adapter")
    else:
        model = attach(None, trainable=False)
        prev = args.output / "initial_adapter"
        model.save_pretrained(prev)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    for rnd in range(args.rounds + 1):
        folder = args.output / f"round_{rnd}"
        folder.mkdir(exist_ok=False)
        teacher = attach(prev, trainable=False)
        _save_json(
            folder / "evaluation.json", _evaluate(teacher, tokenizer, ev, args, rnd)
        )
        if rnd == args.rounds or args.arm == "no_update":
            del teacher
            gc.collect()
            torch.cuda.empty_cache()
            continue
        student = attach(prev, trainable=True)
        student.train()
        optim = torch.optim.AdamW(
            [p for p in student.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=0.0,
        )
        records = []
        with (folder / "training.jsonl").open("x") as output:
            for row in train:
                record = _train_source(
                    teacher,
                    student,
                    optim,
                    tokenizer,
                    row,
                    audit[row["source_id"]]["calibration_tests"],
                    args,
                    rnd,
                )
                records.append(record)
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()
        updated = folder / "adapter"
        student.save_pretrained(updated)
        _save_json(
            folder / "summary.json",
            {
                "round": rnd,
                "sources": len(records),
                "optimizer_steps": sum(r.get("step", False) for r in records),
                "abstention_rate": sum(not r.get("step", False) for r in records)
                / len(records),
                "selected_masks": [r.get("selected_mask", 0) for r in records],
            },
        )
        del student, teacher, optim
        gc.collect()
        torch.cuda.empty_cache()
        prev = updated
    print(
        json.dumps(
            {"status": "finished", "arm": args.arm, "output": str(args.output)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
