#!/usr/bin/env python3
"""Five-round EESD learning-credit experiment (prospective, frozen protocol).

Causal-scientific question: does Shapley selection of partial test feedback
transfer through real LoRA updates to correctness on unseen PROGRAMS, beyond
direct exhaustive J(S) maximization, random feedback, and matching the
number of optimizer updates? One model (Qwen2.5-Coder-7B), one benchmark
(RunBugRun Python), 4 feedback tests / 16 exact subsets / G=8 programs.
Requires a strong isolation sandbox. Never trains on meta/final test outcomes.
"""

from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from collections import Counter

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
from eesd.learning_credit_selection import (
    selection_from_arm, scoring_cost,
)

ARMS = ("no_update", "sdpo", "audit_gate", "direct_max", "shapley",
        "random_subset", "matched_random_updates")


def _digest_adapter(path: Path) -> str:
    """Bind all LoRA checkpoint bytes to every experiment arm."""
    files=sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError("empty initial adapter")
    h=hashlib.sha256()
    for p in files:
        rel=p.relative_to(path).as_posix().encode()
        h.update(len(rel).to_bytes(4,"big"))
        h.update(rel)
        with p.open("rb") as file:
            for chunk in iter(lambda:file.read(1024*1024),b""):
                h.update(chunk)
    return h.hexdigest()


def _load_schedule(path: Path, args, manifest_sha: str, adapter_sha: str,
                   source_ids: list[str]) -> dict | None:
    if args.arm not in ("random_subset","matched_random_updates"):
        if path is not None:
            raise ValueError("schedule is only for the two frozen random controls")
        return None
    if path is None or not path.is_file():
        raise ValueError("matched random arms require frozen --schedule from prior Shapley run")
    receipt=json.loads(path.read_text())
    if (receipt.get("schema")!="eesd-learning-credit-schedule-v1"
        or receipt.get("seed")!=args.seed or receipt.get("rounds")!=args.rounds
        or receipt.get("data_manifest_sha256")!=manifest_sha
        or receipt.get("initial_adapter_sha256")!=adapter_sha
        or receipt.get("source_ids")!=sorted(source_ids)):
        raise ValueError("schedule protocol/source/seed/adapter mismatches")
    if len(receipt["training_rounds"])!=args.rounds:
        raise ValueError("schedule missing rounds")
    population=set(source_ids)
    for k,entry in enumerate(receipt["training_rounds"]):
        masks=entry.get("accepted_masks")
        random_ids=entry.get("random_update_sources")
        target=entry.get("shapley_actual_optimizer_steps")
        if (entry.get("round")!=k or not isinstance(masks,dict)
            or set(masks)!=population
            or any(type(v) is not int or not 0<=v<=15 for v in masks.values())
            or type(target) is not int or target!=sum(bool(v) for v in masks.values())
            or not isinstance(random_ids,list) or len(random_ids)!=target
            or len(random_ids)!=len(set(random_ids))
            or not set(random_ids)<=population):
            raise ValueError("malformed training-only reference update schedule")
    return receipt



def load_matched_data(data_dir: Path, train_limit: int, eval_limit: int):
    train, eval_rows, manifest = legacy.read_dataset(data_dir, train_limit, eval_limit)
    if manifest.get("behavioral_schema") != "eesd-runbugrun-behavioral-v1":
        raise ValueError("behavioral RunBugRun train/feedback/calibration/audit split required")
    n_cal=int(manifest.get("calibration_tests",4))
    n_audit=int(manifest.get("audit_tests",4))
    if n_cal<2 or n_audit<2:
        raise ValueError("insufficient calibration / audit test evidence")
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
        if (len(probe["calibration_tests"])!=n_cal
            or len(heldout["audit_tests"])!=n_audit):
            raise ValueError("actual calibration/audit counts differ from sealed manifest")
        pub = {x["id"] for x in row["public_tests"]}
        cal = {x["id"] for x in probe["calibration_tests"]}
        aud = {x["id"] for x in heldout["audit_tests"]}
        if (len(pub)!=4 or len(cal)!=n_cal or len(aud)!=n_audit
            or pub & cal or pub & aud or cal & aud):
            raise ValueError("feedback/calibration/audit test contamination")
        if manifest.get("formal_protocol") == "eesd-nested-T4-vs-T8-20261101":
            sequences=(row["public_tests"],probe["calibration_tests"],
                       heldout["audit_tests"])
            inputs=[str(t["input"]) for group in sequences for t in group]
            if len(set(inputs)) != 4+n_cal+n_audit:
                raise ValueError("formal feedback/calibration/audit input overlap")
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


def _train_source(teacher, student, optimizer, tokenizer, row, calibration_tests,
                  args, round_idx, schedule_round=None):
    """One shared rollout/test budget, independent selection, one LoRA step at most.

    Note: reference-schedule random controls can have fewer actual steps due to
    changed candidates. Report realized steps; do not assert exact matching
    merely from the requested schedule.
    """
    import torch

    source=row["source_id"]
    reference_mask=None
    random_sources=None
    if schedule_round is not None:
        reference_mask=schedule_round["accepted_masks"].get(source)
        random_sources=set(schedule_round["random_update_sources"])
        if args.arm=="random_subset" and reference_mask==0:
            return {"source_id":source,"step":False,"skip":"reference_abstained",
                    "selected_mask":0,"scheduled_step":False}
        if args.arm=="matched_random_updates" and source not in random_sources:
            return {"source_id":source,"step":False,"skip":"random_count_schedule_abstained",
                    "selected_mask":0,"scheduled_step":False}
    sampled=legacy._sample_rollouts(teacher,tokenizer,row,args,round_idx)
    clean=[]
    for _,code in sampled:
        tok=_normalized_ids(tokenizer,code,args.max_new_tokens)
        if tok:
            clean.append((tok,code))
    if len(clean)<2:
        return {"source_id":source,"step":False,"skip":"too_few_valid_programs",
                "selected_mask":0,"scheduled_step":args.arm in ("random_subset","matched_random_updates")}
    feedback=[feedback_for_tests(code,row["public_tests"]) for _,code in clean]
    # The SAME calibration budget is executed for all arms, even sdpo.
    calibration=np.asarray([audit_vector(code,calibration_tests)
                            for _,code in clean],dtype=np.float64)
    score_arm="shapley_select" if args.arm in (
        "shapley","direct_max","random_subset") else "audit_gate"
    scores=[score_candidate(teacher,tokenizer,row,tokens,fb,arm=score_arm,
               max_prompt_tokens=args.max_prompt_tokens)
            for (tokens,_),fb in zip(clean,feedback)]

    choice=selection_from_arm(args.arm,scores,calibration,
        round_index=round_idx,source_id=source,seed=args.seed,
        reference_mask=reference_mask,reference_update_sources=random_sources,
        uncertainty=args.audit_uncertainty,subset_cost=args.subset_cost,
        min_gain=args.min_gain)
    mask=choice.mask
    if args.arm in ("sdpo","matched_random_updates"):
        weights=[np.ones(len(tok),dtype=np.float32) for tok,_ in clean]
    else:
        weights=directional_token_weights(scores,calibration,mask,
                                          shift_scale=args.shift_scale)
    diagnostics={
        "source_id":source, "round":round_idx, "selected_mask":mask,
        "calibration_pass_fraction":float(calibration.mean()),
        "calibration_behavioral_contrast":float(calibration.var(axis=0).mean()),
        "public_pass_fraction":sum(sum(f["outcome"]=="PASS" for f in fb)
                                   for fb in feedback)/(4*len(feedback)),
        "choice_rule":choice.rule,
        "j_calibration":choice.j_calibration,
        "shapley_phi":list(choice.phi) if choice.phi is not None else None,
        "scheduled_step":bool(reference_mask) if args.arm=="random_subset" else
                         source in random_sources if args.arm=="matched_random_updates" else None,
        "candidate_sources":[code for _,code in clean],
        "candidate_shift_means":[(v.mean(axis=1)-v[0].mean()).tolist() for v in scores],
        "candidate_calibration_pass_fraction":calibration.mean(axis=1).tolist(),
        "candidate_code_sha256":[hashlib.sha256(code.encode()).hexdigest()
                                  for _,code in clean],
    }
    if mask==0 or sum(float(w.sum()) for w in weights)<=1e-8:
        return {**diagnostics,"step":False,"skip":
                "abstained_feedback" if mask==0 else "no_aligned_tokens"}
    optimizer.zero_grad(set_to_none=True)
    stats=[]
    count=0
    for (tokens,_),fb,w in zip(clean,feedback,weights):
        if not np.any(w>0):continue
        tq,oldq=masked_teacher(teacher,tokenizer,row,tokens,fb,mask)
        context=legacy.prompt_ids(tokenizer,row)
        inp=torch.tensor([context+tokens],dtype=torch.long,device=student.device)
        loss,stat=distill_token_kl(student,oldq,tq,w,inp,len(context),beta=args.beta)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite student update")
        (loss/len(clean)).backward()
        stats.append(stat);count+=1
        del loss,inp,oldq,tq
    if count==0:
        return {**diagnostics,"step":False,"skip":"zero_token_weight"}
    norm=torch.nn.utils.clip_grad_norm_([p for p in student.parameters()
                                          if p.requires_grad],1.0)
    if not torch.isfinite(norm):raise RuntimeError("nonfinite gradient norm")
    if float(norm.detach().cpu()) <= 1e-10:
        return {**diagnostics,"step":False,"skip":"zero_parameter_gradient"}
    optimizer.step()
    return {**diagnostics,"step":True,"gradient_norm":float(norm.cpu()),
            "mean_weight":float(np.mean([v["mean_weight"] for v in stats])),
            "distilled_candidates":count}


def _save_json(path: Path, data: dict):
    with path.open("x") as output:
        json.dump(data, output, sort_keys=True, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument(
        "--model-config", type=Path, default=ROOT / "rlvr/models/qwen25_7b.yaml"
    )
    ap.add_argument("--init-adapter", type=Path, required=True, help="shared immutable LoRA baseline for every arm")
    ap.add_argument("--schedule", type=Path, help="training-only frozen Shapley schedule for random controls")
    ap.add_argument("--seed", type=int, default=1701)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--train-limit", type=int, default=64)
    ap.add_argument("--eval-limit", type=int, default=128)
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
    if (args.rounds != 5 or args.rollouts < 2 or args.generate_batch < 1
            or args.beta < 0 or not args.init_adapter.is_dir()
            or not (args.init_adapter/"adapter_config.json").is_file()):
        raise ValueError("requires exactly five rounds, >=2 rollouts, and immutable initial adapter")
    if manifest.get("learning_credit_schema")!="eesd-five-round-frozen-data-v1":
        raise ValueError("must use new disjoint train/meta/final dataset")
    # Only validate metadata and immutable checksum. Never read meta/final
    # outcomes into any selector. The evaluation code reads final tests ONLY
    # when reporting each updated checkpoint, not while deciding updates.
    meta_path=args.data_dir/"meta_probes_do_not_train.jsonl"
    if not meta_path.is_file() or "meta_probes_do_not_train.jsonl" not in manifest["file_sha256"]:
        raise ValueError("independent intervention meta set is required")
    manifest_sha=legacy.digest(args.data_dir/"manifest.json")
    adapter_sha=_digest_adapter(args.init_adapter)
    schedule=_load_schedule(args.schedule,args,manifest_sha,adapter_sha,
                            [r["source_id"] for r in train])
    plan = {
        "schema": "eesd-learning-credit-five-v1",
        "status": "preflight_not_run",
        "arm": args.arm,
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "revision": legacy.MODEL_REVISION,
        "train_sources": len(train),
        "eval_sources": len(ev),
        "dataset_manifest_sha256": manifest_sha,
        "initial_adapter_sha256": adapter_sha,
        "meta_probes_file_sha256": manifest["file_sha256"]["meta_probes_do_not_train.jsonl"],
        "meta_sources": manifest["meta_sources"],
        "scoring_cost":scoring_cost(4,args.rollouts,len(train),args.rounds,args.arm),
        "schedule_sha256":legacy.digest(args.schedule) if args.schedule else None,
        "calibration_budget_per_rollout": len(
            next(iter(audit.values()))["calibration_tests"]
        ),
        "reserved_train_audit_tests_per_source": int(manifest.get("audit_tests",4)),
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
    prev = args.init_adapter
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
        torch.cuda.reset_peak_memory_stats()
        train_start_time = time.perf_counter()
        records = []
        schedule_round = schedule["training_rounds"][rnd] if schedule else None
        train_order = list(train)
        requested_steps = None
        if args.arm == "matched_random_updates":
            requested_steps = schedule_round["shapley_actual_optimizer_steps"]
            preselected = set(schedule_round["random_update_sources"])
            first = [r for r in train_order if r["source_id"] in preselected]
            # Stable fallback ranking is chosen BEFORE seeing candidates,
            # independent of evaluation/test performance.
            backup = sorted((r for r in train_order if r["source_id"] not in preselected),
                key=lambda r: hashlib.sha256(
                    f'{args.seed}:{rnd}:{r["source_id"]}:backup'.encode()).hexdigest())
            train_order = first + backup
        actual_steps = 0
        with (folder / "training.jsonl").open("x") as output:
            for row in train_order:
                active_schedule = schedule_round
                if args.arm == "matched_random_updates":
                    if actual_steps >= requested_steps:
                        record={"source_id":row["source_id"],"round":rnd,
                                "step":False,"selected_mask":0,
                                "skip":"exact_update_count_satisfied"}
                        records.append(record)
                        output.write(json.dumps(record,sort_keys=True)+"\n")
                        output.flush()
                        continue
                    # The random source preference was preregistered in the
                    # schedule. Fallback sources are predetermined and used
                    # ONLY if a prior candidate could not produce a real step.
                    active_schedule = dict(schedule_round,
                        random_update_sources=[row["source_id"]])
                record = _train_source(
                    teacher,student,optim,tokenizer,row,
                    audit[row["source_id"]]["calibration_tests"],
                    args,rnd,active_schedule)
                if args.arm == "matched_random_updates":
                    record["preselected_random_source"] = (
                         row["source_id"] in set(schedule_round["random_update_sources"]))
                    record["fallback_random_source"] = not record["preselected_random_source"]
                actual_steps+=int(record.get("step",False))
                records.append(record)
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()
        if requested_steps is not None and actual_steps != requested_steps:
            raise RuntimeError(f"unable to achieve strictly matched actual updates: "
                               f"target={requested_steps}, realized={actual_steps}")
        train_wallclock_s = time.perf_counter()-train_start_time
        gpu_peak_bytes = int(torch.cuda.max_memory_allocated())
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
                "teacher_subset_scoring_calls": int(sum(
                  len(r.get("candidate_sources", []))
                  * (16 if args.arm in ("shapley","direct_max","random_subset")
                     else 2) for r in records)),
                "public_test_executions":int(sum(4*len(r.get("candidate_sources",[]))
                                                for r in records)),
                "calibration_test_executions":int(sum(
                  int(manifest.get("calibration_tests",4))*len(r.get("candidate_sources",[]))
                  for r in records)),
                "training_wallclock_seconds":train_wallclock_s,
                "peak_cuda_allocated_bytes":gpu_peak_bytes,
                "choice_reasons":dict(Counter(r.get("choice_rule",r.get("skip","unknown")) for r in records)),
                "scheduled_optimizer_steps":(
                    schedule["training_rounds"][rnd]["shapley_actual_optimizer_steps"]
                    if schedule else None
                ),
                "realized_steps_equal_schedule":(
                    sum(r.get("step",False) for r in records)
                    == schedule["training_rounds"][rnd]["shapley_actual_optimizer_steps"]
                    if schedule else None
                ),
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
