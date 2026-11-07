#!/usr/bin/env python3
"""Strict six-arm training-parity verifier (no held-out performance inference)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eesd.training_outcomes import tree_sha

ARM_MODES={
    "equal_uniform":"uniform",
    "fixed_uniform":"uniform",
    "adaptive_uniform":"uniform",
    "fixed_edit":"edit",
    "adaptive_edit":"edit",
    "adaptive_random":"random"
}
COMMON=(
    "model_id","revision","model_config_sha256","previous_adapter",
    "seed","response_token_budget","response_tokens","optimizer_steps",
    "micro_steps","training_outcome_protocol",
    "gradient_accumulation","max_length","learning_rate",
    "anchor_beta","lora_r","lora_alpha","trainer_source_sha256",
    "all_trajectories","development_trajectories",
    "scheduled_training_trajectories","sources",
    "include_zero_weight_kl","token_edit_boost"
)
COMMON_AUDIT=("rows","max_prompt_tokens","response_truncated_rows",
              "max_length","sealed_generation_message_rows")


def verify(root:Path)->dict:
    reports={}
    for arm,mode in ARM_MODES.items():
        p=root/arm/"training-report.json"
        if not p.is_file():raise FileNotFoundError(p)
        report=json.loads(p.read_text())
        if report.get("rule")!="eesd_full" or report.get("token_weighting")!=mode:
            raise ValueError(f"{arm}: wrong scientific token-mode or anchor execution slot")
        if report.get("anchor_beta")!=0.03 or not report.get("include_zero_weight_kl"):
            raise ValueError(f"{arm}: KL not matched to submitted objective")
        if report.get("token_edit_boost")!=2.:
            raise ValueError(f"{arm}: token edit boost is not prelocked 2.0")
        if not (root/arm/"adapter"/"adapter_config.json").is_file():
            raise ValueError(f"{arm}: completed LoRA adapter missing")
        reports[arm]=report

    baseline=reports["adaptive_uniform"]
    for arm,report in reports.items():
        for key in COMMON:
            if key not in baseline or report.get(key)!=baseline[key]:
                raise ValueError(f"{arm}: unmatched training parameter {key}")
        for key in COMMON_AUDIT:
            if (report.get("tokenization_audit") or {}).get(key)!=(
                 baseline.get("tokenization_audit") or {}).get(key):
                raise ValueError(f"{arm}: unmatched tokenizer/population field {key}")
    if baseline["response_token_budget"]!=baseline["response_tokens"]:
        raise ValueError("actual response-token budget not met")
    init=baseline.get("previous_adapter")
    if not isinstance(init,str) or not Path(init).is_dir():
        raise ValueError("shared initial adapter path absent")
    if baseline["optimizer_steps"]<1:
        raise ValueError("no actual LoRA training")
    return {
        "schema":"submitted-eed-token-factorial-training-parity-v1",
        "status":"training_parity_verified_only",
        "arms":list(ARM_MODES),
        "seed":baseline["seed"],
        "shared_initial_adapter":init,
        "shared_initial_adapter_tree_sha256":tree_sha(Path(init)),
        "train_response_tokens":baseline["response_tokens"],
        "train_optimizer_steps":baseline["optimizer_steps"],
        "beta":baseline["anchor_beta"],
        "token_edit_boost":baseline["token_edit_boost"],
        "model_id":baseline["model_id"],
        "revision":baseline["revision"],
        "train_source_count":baseline["sources"],
        "no_heldout_scores_computed":True,
        "note":"All six arms match training budget. Independently evaluate saved adapters before asserting correctness gains."
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("parity receipt exists")
    report=verify(a.root)
    a.output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps(report,sort_keys=True,indent=2))


if __name__=="__main__":main()
