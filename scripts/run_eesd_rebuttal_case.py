#!/usr/bin/env python3
"""Independent, immutable ICLR 2027 EED rebuttal case launcher.

Every CASE is executed separately and saved under
  <OUT_ROOT>/<STUDY>/<CASE_ID>/seed<SEED>/
with exact source/config/initial-adapter/runner hashes. No other case is
executed as a side effect. 'prepare' makes training inputs and a sealed
launcher receipt; 'run' consumes exactly that prepared receipt; 'both'
does both. The held-out evaluator must still be invoked independently.

This runner ONLY trains the SUBMITTED 4-state Dirichlet weight, or explicitly
labeled controls. The trainer 'eesd_full' label is the shared anchored loss
*execution slot* rather than proof that the new 3-state scorer was used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import random
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/"scripts"
sys.path.insert(0,str(SCRIPTS))

from eesd.distillation import TRAIN_RULES,validate_scored_record
from eesd.training_outcomes import tree_sha
import prepare_submitted_dirichlet_rebuttal_weights as weightlib
import audit_eesd_token_aware_inputs as input_audit
import check_eesd_rebuttal_coverage as cell_coverage

REGISTRY=ROOT/"configs/rebuttal/eesd_case_registry.json"
TRAINER=ROOT/"scripts/run_eesd_weighted_sft.py"
TOKEN_CODE=ROOT/"src/eesd/token_aware.py"
SAFE=re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def file_sha(p:Path)->str:
    return weightlib.sha(p)


def time_now()->str:
    return datetime.now(timezone.utc).isoformat()


def load_cases()->dict:
    data=json.loads(REGISTRY.read_text())
    if data.get("schema")!="eesd-iclr2027-reviewer-case-registry-v1":
        raise ValueError("incorrect case registry")
    cases={r["id"]:r for r in data["cases"]}
    if len(cases)!=len(data["cases"]):
        raise ValueError("duplicate registered case")
    return cases


def safe_path(root:Path,study:str,case_id:str,seed:int)->Path:
    if not SAFE.fullmatch(study) or not SAFE.fullmatch(case_id):
        raise ValueError("study/case names are directory-safe identifiers")
    if type(seed)is not int or seed<0:
        raise ValueError("seed must be nonnegative")
    return root/study/case_id/f"seed{seed}"


def _save(path:Path,data:dict,*,overwrite=False)->None:
    content=json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if not overwrite:
        with path.open("x") as file:file.write(content)
    else:
        # Only a case state receipt may be updated; never overwrite inputs.
        partial=path.with_name(path.name+".partial")
        partial.write_text(content)
        partial.replace(path)


def training_setup(args,case:dict)->dict:
    if args.corrections is None or args.model_config is None:
        raise ValueError("training case requires --corrections and --model-config")
    if args.init_adapter is None or not (args.init_adapter/"adapter_config.json").is_file():
        raise ValueError("training case requires a real common --init-adapter")
    if args.budget is None or args.budget<1:
        raise ValueError("training case requires positive explicit --budget")
    # Include original EED's whole-policy KL beta and original alpha/lambda.
    cfg=yaml.safe_load(args.model_config.read_text())
    if not isinstance(cfg,dict) or not isinstance(cfg.get("model_id"),str) or (
        not isinstance(cfg.get("revision"),str) or len(cfg["revision"])!=40):
        raise ValueError("pinned model_id and 40-char revision required")
    return {
        "corrections_path":str(args.corrections.resolve()),
        "corrections_sha256":file_sha(args.corrections),
        "model_config_path":str(args.model_config.resolve()),
        "model_config_sha256":file_sha(args.model_config),
        "model_id":cfg["model_id"],"revision":cfg["revision"],
        "shared_init_adapter_path":str(args.init_adapter.resolve()),
        "shared_init_adapter_tree_sha256":tree_sha(args.init_adapter),
        "response_token_budget":args.budget,
        "seed":args.seed,
        "alpha":0.1,"lambda_penalty":0.5,"beta":0.03,
        "token_edit_boost":2.0,
        "trainer_source_sha256":file_sha(TRAINER),
        "scorer_source_sha256":file_sha(SCRIPTS/"prepare_submitted_dirichlet_rebuttal_weights.py"),
        "token_source_sha256":file_sha(TOKEN_CODE),
        "registry_sha256":file_sha(REGISTRY),
    }


def check_source_integrity(setup:dict)->None:
    checks=[
      (Path(setup["corrections_path"]),setup["corrections_sha256"],file_sha),
      (Path(setup["model_config_path"]),setup["model_config_sha256"],file_sha),
      (Path(setup["shared_init_adapter_path"]),setup["shared_init_adapter_tree_sha256"],tree_sha),
      (TRAINER,setup["trainer_source_sha256"],file_sha),
      (SCRIPTS/"prepare_submitted_dirichlet_rebuttal_weights.py",
        setup["scorer_source_sha256"],file_sha),
      (TOKEN_CODE,setup["token_source_sha256"],file_sha),
      (REGISTRY,setup["registry_sha256"],file_sha)
    ]
    for path,expected,fn in checks:
        if not path.exists() or fn(path)!=expected:
            raise ValueError(f"FROZEN input/code checksum changed: {path}")


def input_provenance(args,case:dict)->dict:
    if args.corrections is None:
        return {"status":"unavailable_original_correction_bank"}
    audit=input_audit.audit(args.corrections,eval_ids=args.eval_ids)
    if case["token"] in ("edit","random") and not audit["can_run_complete_edit_ablation"]:
        raise ValueError("edit/random case ineligible: original a0 absent in correction bank")
    if args.require_problem_disjoint:
        evaluation=audit.get("evaluation")
        if not evaluation or not evaluation.get("problem_level_overlap_checked") or (
            any(audit["missing_problem_id_counts"].values())):
            raise ValueError("strict source-independent evaluation requires problem IDs for EVERY correction and evaluator")
    audit["provenance_status"]=(
        "problem_ids_disjoint_checked" if (audit.get("evaluation") or {}).get(
            "problem_level_overlap_checked",False) else
        "unverified_problem_id_disjointness"
    )
    return audit


def make_weighted_training_input(rows:list[dict],case:dict,seed:int):
    """Return original 4-state score receipt and common KL execution-slot rows."""
    kappa=float(case["kappa"])
    mode=case.get("relevance_mode","original")
    if mode not in ("original","uniform","permuted"):
        raise ValueError("unknown relevance intervention")
    if mode!="original" and kappa!=16.:
        raise ValueError("relevance interventions and temperature changes are separate experiments")
    prepared=[]
    changed=0
    for row in rows:
        current=dict(row)
        if mode=="uniform":
            current["relevance"]=[1.0]*len(row["relevance"])
        elif mode=="permuted":
            original=list(row["relevance"])
            if len(original)<2:raise ValueError("cannot test relevance alignment with <2 tests")
            h=hashlib.sha256(f'{seed}:{row["trajectory_id"]}:relevance-permutation'.encode()).digest()
            rng=random.Random(int.from_bytes(h[:8],"big"))
            indices=list(range(len(original)))
            rng.shuffle(indices)
            if indices==list(range(len(original))):
                indices=indices[1:]+indices[:1]
            current["relevance"]=[original[i] for i in indices]
            changed+=int(current["relevance"]!=original)
        prepared.append(current)
    scored,summary=weightlib.add_weights(
        prepared,alpha=.1,lam=.5,seed=seed,include_kappa_sweep=False,kappa=kappa
    )
    summary["relevance_mode"]=mode
    summary["permuted_rows_with_changed_values"]=changed
    summary["original_relevance_preserved_in_source_sha256"]=True
    result=[]
    for original in scored:
        value=original["rebuttal_weights"][case["weight"]]
        slots={name:0.0 for name in TRAIN_RULES}
        slots["eesd_full"]=float(value)
        r={**original,
            "training_weights":slots,
            "manuscript_rebuttal_arm":case["id"],
            "manuscript_weight_name":case["weight"],
            "training_slot":"eesd_full"}
        validate_scored_record(r)
        result.append(r)
    summary["case_id"]=case["id"]
    return result,summary


def _diag_stats(rows:list[dict])->dict:
    from eesd.evidence import effective_mass
    output={"row_count":len(rows),"by_split":{},"kernel_temperatures":[8,16,32],
        "note":"These are public TRAIN/DEV diagnostic measurements, NOT unseen prediction accuracy"}
    for split in ("train","development"):
        sub=[r for r in rows if r["split"]==split]
        stats={}
        for kappa in (8,16,32):
            values=[];masses=[];quality=[]
            for r in sub:
                before=weightlib.to_codes(r,"before_outcomes")
                after=weightlib.to_codes(r,"after_outcomes")
                relevance=r["relevance"] if kappa==16 else weightlib.pow_relevance(
                    r["relevance"],kappa/16)
                score=weightlib.weights_for_record(before,after,relevance)
                values.append(score["adaptive_eed"])
                masses.append(score["effective_mass"])
                quality.append(score["absolute_final_public_pass_fraction"])
            stats[str(kappa)]={
                "n":len(values),
                "mean_effective_mass":sum(masses)/len(masses) if masses else None,
                "mean_correction_ce_weight":sum(values)/len(values) if values else None,
                "zero_ce_weight_fraction":sum(x==0 for x in values)/len(values) if values else None,
                "mean_final_public_pass_fraction":sum(quality)/len(quality) if quality else None,
                "mean_edit_weight_when_nonzero":(
                    sum(x for x in values if x>0)/sum(x>0 for x in values)
                    if any(x>0 for x in values) else None),
            }
        output["by_split"][split]=stats
    return output


def prepare_case(args,case:dict,folder:Path)->dict:
    if folder.exists():
        raise FileExistsError(f"immutable case directory already exists: {folder}")
    kind=case["kind"]
    if kind in ("train","baseline") and args.init_adapter is None:
        raise ValueError("baseline/train cases require --init-adapter")
    if args.corrections is None and kind not in ("prediction",):
        raise ValueError("input correction JSONL required")
    setup=training_setup(args,case) if kind=="train" else {}
    provenance=input_provenance(args,case) if args.corrections is not None else {
        "status":"no_corrections_supplied"}
    if kind=="baseline":
        if not (args.init_adapter/"adapter_config.json").is_file():
            raise ValueError("frozen baseline requires real initial adapter")
        setup={
            "shared_init_adapter_path":str(args.init_adapter.resolve()),
            "shared_init_adapter_tree_sha256":tree_sha(args.init_adapter),
            "model_config_path":str(args.model_config.resolve()) if args.model_config else None,
            "model_config_sha256":file_sha(args.model_config) if args.model_config else None,
            "source":"no training; no downstream evaluation implied"
        }
    folder.mkdir(parents=True,exist_ok=False)
    _save(folder/"data_provenance.json",provenance)
    file_inputs={}
    command=None
    if kind=="train":
        rows=[json.loads(x) for x in args.corrections.read_text().split("\n") if x.strip()]
        scored,summary=make_weighted_training_input(rows,case,args.seed)
        filename=folder/"scored-corrections.jsonl"
        with filename.open("x") as file:
            for r in scored:
                file.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n")
        _save(folder/"weight_diagnostics.json",summary)
        file_inputs["scored-corrections.jsonl"]=file_sha(filename)
        command=[
            sys.executable,str(TRAINER),
            "--input",str(filename.resolve()),
            "--model-config",setup["model_config_path"],
            "--rule","eesd_full","--include-zero-weight-kl",
            "--token-weighting",case["token"],
            "--token-edit-boost",str(setup["token_edit_boost"]),
            "--previous-adapter",setup["shared_init_adapter_path"],
            "--seed",str(args.seed),
            "--response-token-budget",str(setup["response_token_budget"]),
            "--anchor-beta",str(setup["beta"]),
            "--output",str((folder/"model").resolve())
        ]
    elif kind=="diagnostic":
        rows=[json.loads(x) for x in args.corrections.read_text().split("\n") if x.strip()]
        _save(folder/"relevance_diagnostic.json",_diag_stats(rows))
    elif kind=="descriptor":
        import audit_eesd_original_descriptor as descriptor
        report=descriptor.audit(args.corrections,pbpf_root=getattr(args,"pbpf_root",None))
        _save(folder/"original_descriptor_audit.json",report)
    elif kind=="prediction":
        # Prediction raw pairs require a separate explicit future-label audit.
        # This preflight never synthesizes labels or quietly copies train labels.
        external=args.prediction_rows
        if external is None:
            _save(folder/"prediction_status.json",{
                "status":"blocked_original_future_outcome_records_missing",
                "reason":"The GitHub paper summary has only aggregate NLL, not the 3,000 raw prediction labels/probabilities."})
        else:
            import audit_eesd_future_outcome_pairs as predictor
            report=predictor.summarize(external)
            _save(folder/"prediction_pair_audit.json",report)
            file_inputs["prediction_file_sha256"]=file_sha(external)
    elif kind=="audit":
        pass
    elif kind!="baseline":
        raise ValueError("unknown case kind")
    if args.eval_ids is not None:file_inputs["eval_ids_sha256"]=file_sha(args.eval_ids)
    manifest={
        "schema":"eesd-iclr2027-independent-case-v1",
        "case":case,"study":args.study,"seed":args.seed,
        "original_paper_cell":getattr(args,"cell",None),
        "cell_identity_audit":getattr(args,"cell_identity",None),
        "created_at":time_now(),"status":"prepared",
        "setup":setup,"frozen_case_input_hashes":file_inputs,
        "data_provenance_sha256":file_sha(folder/"data_provenance.json"),
        "command":command,
        "claims":"NONE: case preparation does not establish correctness or reviewer acceptance",
        "heldout_evaluation":"independent sealed generation+execution mandatory"
    }
    _save(folder/"case.json",manifest)
    return manifest


def run_case(folder:Path)->dict:
    record=folder/"case.json"
    if not record.is_file():
        raise FileNotFoundError(f"prepare first: {folder}")
    manifest=json.loads(record.read_text())
    if manifest.get("schema")!="eesd-iclr2027-independent-case-v1":
        raise ValueError("invalid case receipt")
    if manifest.get("status")!="prepared":
        raise ValueError("case is not in prepared state (no overwrite/retry)")
    kind=manifest["case"]["kind"]
    if kind!="train":
        if kind=="prediction" and (folder/"prediction_status.json").is_file():
            manifest["status"]="blocked_missing_original_future_outcome_records"
        else:
            manifest["status"]="completed_diagnostic_without_gpu"
        manifest["finished_at"]=time_now()
        _save(record,manifest,overwrite=True)
        return manifest
    setup=manifest["setup"]
    check_source_integrity(setup)
    filename=folder/"scored-corrections.jsonl"
    if file_sha(filename)!=manifest["frozen_case_input_hashes"]["scored-corrections.jsonl"]:
        raise ValueError("case training rows were modified after preparation")
    cmd=manifest["command"]
    if not cmd or Path(cmd[-1])!= (folder/"model").resolve():
        raise ValueError("malformed frozen trainer command")
    if (folder/"model").exists():
        raise FileExistsError("model output already exists; cannot silently resume/retrain")
    manifest.update(status="running",started_at=time_now())
    _save(record,manifest,overwrite=True)
    try:
        with (folder/"training_stdout_stderr.log").open("x") as stream:
            completed=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                     check=False)
        manifest["returncode"]=completed.returncode
        if completed.returncode!=0:
            manifest["status"]="failed"
        else:
            training_report=folder/"model"/"training-report.json"
            nonestimable=folder/"model"/"training-outcome.json"
            if training_report.is_file():
                report=json.loads(training_report.read_text())
                for key,expected in [
                    ("input_sha256",file_sha(filename)),
                    ("model_config_sha256",setup["model_config_sha256"]),
                    ("seed",setup["seed"]),
                    ("response_tokens",setup["response_token_budget"]),
                    ("anchor_beta",setup["beta"]),
                    ("token_weighting",manifest["case"]["token"]),
                ]:
                    if report.get(key)!=expected:
                        raise ValueError(f"TRAINING BINDING ERROR {key}: {report.get(key)} vs {expected}")
                if not (folder/"model"/"adapter"/"adapter_config.json").is_file():
                    raise ValueError("trained LoRA adapter absent")
                manifest["status"]="trained_awaiting_independent_evaluation"
                manifest["training_report_sha256"]=file_sha(training_report)
                manifest["trained_adapter_tree_sha256"]=tree_sha(folder/"model"/"adapter")
            elif nonestimable.is_file():
                manifest["status"]="non_estimable_zero_positive_training_weight"
                manifest["training_outcome_sha256"]=file_sha(nonestimable)
            else:
                raise RuntimeError("trainer exit 0 but no scientific training/outcome receipt")
    except BaseException as e:
        manifest.update(status="failed",error=repr(e))
        _save(record,manifest,overwrite=True)
        raise
    manifest["finished_at"]=time_now()
    _save(record,manifest,overwrite=True)
    if manifest["status"]=="failed":
        raise RuntimeError(f"training failed, preserved full log: {folder/'training_stdout_stderr.log'}")
    return manifest


def main(*, selected_case:str|None=None):
    """One fixed CASE per entrypoint; no CLI override can change its identity."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list",action="store_true",help="list all registered case IDs and reviewer mapping")
    p.add_argument("--case",choices=sorted(load_cases()),
                   help="Case registry ID; omitted for scripts/rebuttal_cases/cXX_*.py")
    p.add_argument("--study",type=str,default="codearc_deepseek")
    p.add_argument("--cell",choices=sorted(cell_coverage.load_matrix()),
                   help="Strongly recommended: auto-name study and bind model+dataset to one of the 12 original paper cells.")
    p.add_argument("--strict-record-identity",action="store_true",
                   help="Require every correction record to state the chosen model revision and dataset.")
    p.add_argument("--seed",type=int,default=1701)
    p.add_argument("--out-root",type=Path,default=Path("runs/eesd-iclr2027-rebuttal"))
    p.add_argument("--action",choices=("prepare","run","both"),default="prepare")
    p.add_argument("--corrections",type=Path)
    p.add_argument("--model-config",type=Path)
    p.add_argument("--init-adapter",type=Path)
    p.add_argument("--budget",type=int)
    p.add_argument("--eval-ids",type=Path)
    p.add_argument("--prediction-rows",type=Path)
    p.add_argument("--pbpf-root",type=Path,help="Optional exact pinned legacy PBPF checkout for original lexical-kernel recomputation")
    p.add_argument("--require-problem-disjoint",action="store_true")
    args=p.parse_args()
    cases=load_cases()
    if args.list:
        print(json.dumps([{"id":c["id"],"kind":c["kind"],"priority":c["priority"],
                           "reviewers":c["reviewers"],"goal":c["goal"]}
                          for c in cases.values()],indent=2,ensure_ascii=False))
        return
    if selected_case is not None:
        if selected_case not in cases:
            p.error("wrapper case is not registered")
        if args.case is not None and args.case!=selected_case:
            p.error("independent case entrypoint cannot run another --case")
        args.case=selected_case
    if args.case is None:p.error("--case is required except for --list")
    if selected_case is not None and args.cell is None:
        p.error("standalone C00-C16/D01-D04 Python files require --cell to bind the original model and dataset")
    if args.cell:
        profile=cell_coverage.load_matrix()[args.cell]
        if args.model_config is not None:
            cfg=yaml.safe_load(args.model_config.read_text())
            if (cfg.get("model_id")!=profile["model_id"] or
                cfg.get("revision")!=profile["model_revision"]):
                p.error("model YAML identity differs from selected --cell")
        if args.action in ("prepare","both") and args.corrections is not None:
            args.cell_identity=cell_coverage.audit_correction_identity(
                args.cell,args.corrections,strict=args.strict_record_identity
            )
        else:
            args.cell_identity=None
        args.study=args.cell
    else:
        args.cell_identity=None
    folder=safe_path(args.out_root,args.study,args.case,args.seed)
    case=cases[args.case]
    if args.action in ("prepare","both"):
        manifest=prepare_case(args,case,folder)
        print(json.dumps({"case":args.case,"status":manifest["status"],
                          "folder":str(folder),"command":manifest["command"]},
                         ensure_ascii=False))
    if args.action in ("run","both"):
        manifest=run_case(folder)
        print(json.dumps({"case":args.case,"status":manifest["status"],
                          "folder":str(folder)},ensure_ascii=False))


if __name__=="__main__":main()
