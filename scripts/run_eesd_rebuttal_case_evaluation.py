#!/usr/bin/env python3
"""Run a SINGLE frozen EESD case through the original PBPF fresh evaluator.

This is NOT a new evaluator. Invokes the pinned yuhanlydia/PBPF
  research/eesd-seed1701-resume-20260922 @
  0200283849274b49e303638b91f142acd8bba984
scripts/generate_apbpf_{rbr,codearc}_bank.py and
scripts/evaluate_eesd_fresh_bank.py, refusing direct-no-sandbox.

Stages:
 --action prepare  : freeze evaluator source+dataset+adapter+commands
 --action run      : generate one greedy fresh program per held-out source,
                     then evaluate in original sandbox; save all receipts.
 --action both     : both, no other CASE is touched.
Output inside the existing case directory:
  evaluation/plan.json, bank/, heldout/report.json, *.log, complete.json

Gemma and replay domains use different evaluators, and are explicitly
NOT supported by this PBPF RunBugRun/CodeARC bridge. Do not mix protocols.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from eesd.training_outcomes import tree_sha

PBPF_PIN="0200283849274b49e303638b91f142acd8bba984"
MODEL_FAMILIES={
  "Qwen/Qwen2.5-Coder-7B-Instruct":"qwen25_7b",
  "deepseek-ai/deepseek-coder-6.7b-instruct":"deepseek_6p7b",
}

def file_sha(p:Path)->str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_once(p:Path,value:dict):
    with p.open("x") as out:
        json.dump(value,out,sort_keys=True,indent=2)
        out.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_checkout(pbpf:Path,domain:str)->dict:
    if domain not in ("rbr","codearc"):raise ValueError("PBPF bridge only supports rbr/codearc")
    if not (pbpf/".git").exists():raise ValueError("PBPF must be a full pinned git checkout")
    head=subprocess.run(["git","rev-parse","HEAD"],cwd=pbpf,text=True,capture_output=True,check=True).stdout.strip()
    if head!=PBPF_PIN:raise ValueError(f"PBPF original evaluator revision mismatch; expected {PBPF_PIN}, got {head}")
    status=subprocess.run(["git","status","--porcelain"],cwd=pbpf,text=True,capture_output=True,check=True).stdout.strip()
    if status:raise ValueError("PBPF checkout is dirty; evaluator code cannot be certified")
    generator=pbpf/"scripts"/f"generate_apbpf_{domain}_bank.py"
    evaluator=pbpf/"scripts"/"evaluate_eesd_fresh_bank.py"
    if not generator.is_file() or not evaluator.is_file():
        raise FileNotFoundError("original PBPF generation/evaluation scripts absent")
    return {"pbpf_commit":head,"generator_script":str(generator),
            "generator_sha256":file_sha(generator),
            "evaluator_script":str(evaluator),
            "evaluator_sha256":file_sha(evaluator)}


def inventory_ids(public:Path,evaluator:Path)->dict:
    for folder in (public,evaluator):
        if not (folder/"tasks.jsonl").is_file() or not (folder/"manifest.json").is_file():
            raise ValueError(f"sealed materialization files missing: {folder}")
    pub=json.loads((public/"manifest.json").read_text())
    ev=json.loads((evaluator/"manifest.json").read_text())
    if pub.get("public_tasks_sha256")!=file_sha(public/"tasks.jsonl"):
        raise ValueError("public source-bank task hash mismatch")
    if ev.get("evaluator_tasks_sha256")!=file_sha(evaluator/"tasks.jsonl"):
        raise ValueError("heldout evaluator task hash mismatch")
    if ev.get("public_tasks_sha256")!=pub["public_tasks_sha256"]:
        raise ValueError("public and evaluator banks have different task origins")
    public_primary=[
        json.loads(s) for s in (public/"tasks.jsonl").read_text().split("\n")
        if s.strip() and json.loads(s).get("split")=="primary"
    ]
    evaluate_primary=[
        json.loads(s) for s in (evaluator/"tasks.jsonl").read_text().split("\n")
        if s.strip() and json.loads(s).get("split")=="primary"
    ]
    actual={str(z["source_component_id"]) for z in evaluate_primary}
    observed={str(z["source_component_id"]) for z in public_primary}
    if not actual or observed!=actual:
        raise ValueError("public/heldout primary source IDs differ")
    return {
        "public_manifest_sha256":file_sha(public/"manifest.json"),
        "public_tasks_sha256":file_sha(public/"tasks.jsonl"),
        "evaluator_manifest_sha256":file_sha(evaluator/"manifest.json"),
        "evaluator_tasks_sha256":file_sha(evaluator/"tasks.jsonl"),
        "evaluation_source_ids":sorted(actual),
        "evaluation_sources":len(actual),
    }


def prepare(case:Path,pbpf:Path,public:Path,evaluator:Path,domain:str,workers:int,
            eval_ids:Path|None=None)->dict:
    case_record=case/"case.json"
    if not case_record.is_file():raise FileNotFoundError("prepare/train independent case first")
    receipt=json.loads(case_record.read_text())
    if receipt.get("schema")!="eesd-iclr2027-independent-case-v1":
        raise ValueError("invalid EED independent case")
    if receipt["status"] not in (
        "trained_awaiting_independent_evaluation","completed_without_gpu"):
        raise ValueError("case has no verified training/baseline adapter")
    if workers<1 or workers>64:raise ValueError("invalid evaluation parallel worker count")
    kind=receipt["case"]["kind"]
    if kind not in ("train","baseline"):
        raise ValueError("diagnostic case has no model to evaluate")
    setup=receipt["setup"]
    model_id=setup.get("model_id")
    if model_id is None and setup.get("model_config_path"):
        import yaml
        model_id=yaml.safe_load(Path(setup["model_config_path"]).read_text()).get("model_id")
    if model_id not in MODEL_FAMILIES:
        raise ValueError("Original PBPF evaluator supports Qwen/DeepSeek only; Gemma/Replay needs separate controlled evaluator")
    family=MODEL_FAMILIES[model_id]
    if kind=="train":
        adapter=case/"model"/"adapter"
        if tree_sha(adapter)!=receipt["trained_adapter_tree_sha256"]:
            raise ValueError("trained adapter differs from case outcome receipt")
        revision=setup["revision"]
    else:
        adapter=Path(setup["shared_init_adapter_path"])
        if tree_sha(adapter)!=setup["shared_init_adapter_tree_sha256"]:
            raise ValueError("baseline initial adapter was changed")
        import yaml
        revision=yaml.safe_load(Path(setup["model_config_path"]).read_text())["revision"]
    pin=validate_checkout(pbpf,domain)
    inv=inventory_ids(public,evaluator)
    sealed_eval_sha=receipt.get("frozen_case_input_hashes",{}).get("eval_ids_sha256")
    if sealed_eval_sha is not None:
        if eval_ids is None or not eval_ids.is_file() or file_sha(eval_ids)!=sealed_eval_sha:
            raise ValueError("evaluation source-ID file must match the original prepared CASE hash")
    if eval_ids is not None:
        rows=[json.loads(z) for z in eval_ids.read_text().split("\n") if z.strip()]
        source_ids=[str(z["source_component_id"]) for z in rows]
        if len(set(source_ids))!=len(source_ids) or sorted(source_ids)!=inv["evaluation_source_ids"]:
            raise ValueError("frozen evaluation source IDs do not match actual PBPF evaluator population")
    output=case/"evaluation"
    if output.exists():raise FileExistsError("case evaluator already prepared")
    output.mkdir(parents=True,exist_ok=False)
    gen=(
        [sys.executable,str(pbpf/"scripts"/f"generate_apbpf_{domain}_bank.py"),
         "--public-root",str(public.resolve()),
         "--output",str((output/"bank").resolve()),
         "--family",family,"--adapter",str(adapter.resolve()),
         "--split","primary","--candidates","1","--greedy",
         "--seed",str(receipt["seed"]),
         "--max-input-tokens","4096",
         "--max-new-tokens",str(1024 if domain=="rbr" else 512)]
    )
    ev=[
        sys.executable,str(pbpf/"scripts"/"evaluate_eesd_fresh_bank.py"),
        "--domain",domain,"--evaluator-root",str(evaluator.resolve()),
        "--bank",str((output/"bank").resolve()),
        "--output",str((output/"heldout").resolve()),
        "--workers",str(workers),
        "--execution-profile","sandbox","--timeout","6.0"
    ]
    plan={
        "schema":"eesd-rebuttal-per-case-original-evaluator-v1",
        "status":"prepared",
        "case_sha256":file_sha(case_record),
        "case_id":receipt["case"]["id"],
        "study":receipt["study"],
        "seed":receipt["seed"],
        "model_id":model_id,"revision":revision,
        "family":family,"domain":domain,
        "adapter_path":str(adapter.resolve()),
        "adapter_tree_sha256":tree_sha(adapter),
        "pbpf_root":str(pbpf.resolve()),
        "public_root":str(public.resolve()),
        "evaluator_root":str(evaluator.resolve()),
        "eval_source_id_file_path":str(eval_ids.resolve()) if eval_ids else None,
        "eval_source_id_file_sha256":file_sha(eval_ids) if eval_ids else None,
        "source_disjointness_status":json.loads((case/"data_provenance.json").read_text()).get(
            "provenance_status","not_certified"
        ),
        **pin,
        **inv,
        "generated_program_count_policy":"one greedy candidate per independent public PRIMARY source",
        "evaluation_policy":"all available tests / original PBPF sandbox worker",
        "generation_command":gen,"heldout_command":ev,
        "scientific_status":"generation/evaluation NOT YET RUN",
        "created_at":now(),
    }
    # Do not leak full train bank or held-out source code into this receipt.
    write_once(output/"plan.json",plan)
    return plan


def _check_frozen(plan,case):
    if file_sha(case/"case.json")!=plan["case_sha256"]:
        raise ValueError("case receipt changed after evaluator prepared")
    if tree_sha(Path(plan["adapter_path"]))!=plan["adapter_tree_sha256"]:
        raise ValueError("model adapter changed after evaluator prepared")
    pbpf=Path(plan["pbpf_root"])
    pin=validate_checkout(pbpf,plan["domain"])
    if any(pin[k]!=plan[k] for k in pin):
        raise ValueError("original generator/evaluator software hash changed")
    inv=inventory_ids(Path(plan["public_root"]),Path(plan["evaluator_root"]))
    if any(inv[k]!=plan[k] for k in inv):
        raise ValueError("heldout source bank changed after evaluator prepared")
    if plan.get("eval_source_id_file_sha256") and (
        not Path(plan["eval_source_id_file_path"]).is_file()
        or file_sha(Path(plan["eval_source_id_file_path"]))!=plan["eval_source_id_file_sha256"]
    ):
        raise ValueError("frozen source ID map changed since evaluator preparation")


def run(case:Path)->dict:
    p=case/"evaluation"/"plan.json"
    if not p.is_file():raise FileNotFoundError("prepare evaluation first")
    plan=json.loads(p.read_text())
    if plan.get("schema")!="eesd-rebuttal-per-case-original-evaluator-v1":
        raise ValueError("not an original PBPF evaluation plan")
    if plan["status"]!="prepared":
        raise ValueError("evaluation already consumed, refuse silent resume")
    _check_frozen(plan,case)
    plan["status"]="running"
    partial=p.with_suffix(".partial")
    partial.write_text(json.dumps(plan,indent=2,sort_keys=True)+"\n")
    partial.replace(p)
    try:
        env=os.environ.copy()
        env["PYTHONPATH"]=str(Path(plan["pbpf_root"])/"src") + os.pathsep + env.get("PYTHONPATH","")
        for label,key in (("generation","generation_command"),("heldout","heldout_command")):
            with (p.parent/(label+".log")).open("x") as log:
                ret=subprocess.run(plan[key],cwd=plan["pbpf_root"],env=env,
                                   stdout=log,stderr=subprocess.STDOUT,check=False)
            if ret.returncode!=0:
                raise RuntimeError(f"{label} exit={ret.returncode}; inspect separate log")
            if label=="generation":
                bank=json.loads((p.parent/"bank"/"run.json").read_text())
                if (bank.get("model")!=plan["model_id"]
                    or bank.get("revision")!=plan["revision"]
                    or bank.get("adapter_sha256")!=plan["adapter_tree_sha256"]
                    or bank.get("public_tasks_sha256")!=plan["public_tasks_sha256"]
                    or bank.get("split")!="primary" or bank.get("candidates")!=1
                    or bank.get("decode_policy")!="greedy"):
                    raise ValueError("generated program ancestry/sampling mismatch")
        report_path=p.parent/"heldout"/"report.json"
        report=json.loads(report_path.read_text())
        if (report.get("schema")!="eesd-fresh-policy-eval-v1"
            or report.get("model")!=plan["model_id"]
            or report.get("revision")!=plan["revision"]
            or report.get("adapter_sha256")!=plan["adapter_tree_sha256"]
            or report.get("task_manifest_sha256")!=plan["evaluator_manifest_sha256"]
            or report.get("sources")!=plan["evaluation_sources"]
            or report.get("execution_profile")!="sandbox"):
            raise ValueError("final heldout evaluator report ancestry mismatch")
        sources=[str(r["source_component_id"]) for r in report["records"]]
        if sorted(sources)!=plan["evaluation_source_ids"] or len(set(sources))!=len(sources):
            raise ValueError("heldout source population does not match sealed inventory")
        plan.update(status="finished_with_real_heldout",
                    finished_at=now(),
                    heldout_report_sha256=file_sha(report_path),
                    heldout_all_tests_pass_at_1=report["fresh_all_tests_pass_at_1"],
                    heldout_count=report["sources"],
                    scientific_status="new model evaluation; require matched arms and source disjointness before comparative claim")
    except BaseException as e:
        plan.update(status="failed",error=repr(e),finished_at=now())
        partial.write_text(json.dumps(plan,indent=2,sort_keys=True)+"\n")
        partial.replace(p)
        raise
    partial.write_text(json.dumps(plan,indent=2,sort_keys=True)+"\n")
    partial.replace(p)
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-dir",type=Path,required=True)
    p.add_argument("--pbpf-root",type=Path)
    p.add_argument("--public-root",type=Path)
    p.add_argument("--evaluator-root",type=Path)
    p.add_argument("--domain",choices=("rbr","codearc"))
    p.add_argument("--workers",type=int,default=12)
    p.add_argument("--eval-ids",type=Path,help="Required when the model case sealed an evaluation ID file; checks population equality")
    p.add_argument("--action",choices=("prepare","run","both"),default="prepare")
    a=p.parse_args()
    case=a.case_dir.resolve()
    if a.action in ("prepare","both"):
        if any(v is None for v in (a.pbpf_root,a.public_root,a.evaluator_root,a.domain)):
            p.error("evaluator preparation requires PBPF checkout, public/evaluator source roots and domain")
        plan=prepare(case,a.pbpf_root.resolve(),a.public_root.resolve(),
                     a.evaluator_root.resolve(),a.domain,a.workers,
                     a.eval_ids.resolve() if a.eval_ids else None)
        print(json.dumps({"status":plan["status"],"case":plan["case_id"],
                          "evaluation_sources":plan["evaluation_sources"]}))
    if a.action in ("run","both"):
        plan=run(case)
        print(json.dumps({"status":plan["status"],"case":plan["case_id"],
                          "pass_at_1":plan["heldout_all_tests_pass_at_1"]}))


if __name__=="__main__":main()
