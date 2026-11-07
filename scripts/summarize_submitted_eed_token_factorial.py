#!/usr/bin/env python3
"""Paired TEST-ONLY evaluation of original EED x edit-token factorial.

Input directory contains six arm files:
    equal_uniform.json, fixed_uniform.json, adaptive_uniform.json,
    fixed_edit.json, adaptive_edit.json, adaptive_random.json

Each must be a real, independently generated eesd-fresh-policy-eval-v1
report with identical frozen source IDs and execution locks. This script
does NOT train, generate, execute, or infer train/eval disjointness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARMS=(
    "equal_uniform","fixed_uniform","adaptive_uniform",
    "fixed_edit","adaptive_edit","adaptive_random"
)
IDENTITY_FIELDS=("domain","model","revision","task_manifest_sha256",
                 "execution_lock_sha256","execution_profile")
CONTRASTS={
    "adaptive_mass_uniform": {"adaptive_uniform":1,"fixed_uniform":-1},
    "adaptive_edit_gain": {"adaptive_edit":1,"adaptive_uniform":-1},
    "fixed_edit_gain": {"fixed_edit":1,"fixed_uniform":-1},
    "edit_mass_interaction": {
        "adaptive_edit":1,"fixed_edit":-1,"adaptive_uniform":-1,"fixed_uniform":1
    },
    "edited_vs_random_same_size": {"adaptive_edit":1,"adaptive_random":-1},
    "adaptive_vs_equal_uniform":{"adaptive_uniform":1,"equal_uniform":-1},
}


def _read(path:Path)->tuple[dict,dict]:
    if not path.is_file():raise FileNotFoundError(path)
    report=json.loads(path.read_text())
    if report.get("schema")!="eesd-fresh-policy-eval-v1":
        raise ValueError("not a frozen fresh-policy evaluation")
    records=report.get("records")
    if not isinstance(records,list) or not records:
        raise ValueError("empty/unstructured evaluation records")
    rows={}
    for r in records:
        sid=r.get("source_component_id")
        if not isinstance(sid,str) or not sid or sid in rows:
            raise ValueError("duplicate or invalid evaluation source ID")
        v=r.get("all_tests_pass")
        if v not in (True,False,0,1) or isinstance(v,str):
            raise ValueError("all_tests_pass must be boolean/binary")
        pid=r.get("problem_id",r.get("task_id"))
        rows[sid]={"success":int(v),"problem_id":str(pid) if pid is not None else None}
    for key in IDENTITY_FIELDS:
        if not isinstance(report.get(key),str) or not report[key]:
            raise ValueError(f"missing required frozen evaluation identity {key}")
    return report,rows


def summarize(eval_dir:Path,*,draws:int=10000,seed:int=314159)->dict:
    if draws<100:raise ValueError("at least 100 bootstrap draws required")
    all_reports={}
    all_rows={}
    for arm in ARMS:
        report,rows=_read(eval_dir/(arm+".json"))
        all_reports[arm]=report
        all_rows[arm]=rows
    baseline=all_reports["adaptive_uniform"]
    identity={key:baseline[key] for key in IDENTITY_FIELDS}
    sourceids=sorted(all_rows["adaptive_uniform"])
    for arm,report in all_reports.items():
        if any(report[key]!=identity[key] for key in IDENTITY_FIELDS):
            raise ValueError(f"mismatched evaluator/benchmark lock for {arm}")
        if sorted(all_rows[arm])!=sourceids:
            raise ValueError(f"evaluation source population differs for {arm}")
    problem_by_source={sid:all_rows["adaptive_uniform"][sid]["problem_id"]
                      for sid in sourceids}
    if any(v is None for v in problem_by_source.values()):
        # Source-ID bootstrap is NOT a reliable problem-cluster CI if multiple
        # bug instances belong to one problem. Require explicit problem IDs.
        raise ValueError("all evaluation records need problem_id or task_id for independent problem-cluster statistics")
    for arm in ARMS:
        for sid in sourceids:
            if all_rows[arm][sid]["problem_id"]!=problem_by_source[sid]:
                raise ValueError(f"problem ID changed for {sid} in {arm}")
    problems=sorted(set(problem_by_source.values()))
    grouped={p:[i for i,sid in enumerate(sourceids) if problem_by_source[sid]==p]
             for p in problems}
    y={arm:np.array([all_rows[arm][sid]["success"] for sid in sourceids],
                    dtype=float) for arm in ARMS}
    problem_rates={arm:np.array([y[arm][grouped[p]].mean() for p in problems])
                   for arm in ARMS}
    rng=np.random.default_rng(seed)
    sample_indices=rng.integers(0,len(problems),size=(draws,len(problems)))
    contrasts={}
    for name,weights in CONTRASTS.items():
        per_problem=sum(mult*problem_rates[arm] for arm,mult in weights.items())
        boot=per_problem[sample_indices].mean(axis=1)
        lo,hi=np.quantile(boot,[.025,.975])
        contrasts[name]={
            "problem_mean_effect":float(per_problem.mean()),
            "problem_cluster_bootstrap_CI95":[float(lo),float(hi)],
            "source_mean_effect":float(sum(mult*y[arm].mean()
                                              for arm,mult in weights.items()))
        }
    pairs={}
    for a,b in (
      ("adaptive_uniform","fixed_uniform"),("adaptive_uniform","equal_uniform"),
      ("adaptive_edit","adaptive_uniform"),("adaptive_edit","fixed_edit"),
      ("adaptive_edit","adaptive_random")
    ):
        key=a+"__vs__"+b
        pairs[key]={
            "fixes":int(((y[a]==1)&(y[b]==0)).sum()),
            "regressions":int(((y[a]==0)&(y[b]==1)).sum()),
            "net_source_fixes":int((y[a]-y[b]).sum())
        }
    return {
        "schema":"submitted-eed-token-factorial-paired-results-v1",
        "eval_identity":identity,
        "source_count":len(sourceids),
        "independent_problem_count":len(problems),
        "arms":{k:{"source_pass_at_1":float(y[k].mean()),
                   "problem_mean_pass_at_1":float(problem_rates[k].mean())}
                for k in ARMS},
        "contrasts":contrasts,
        "paired_fix_regress":pairs,
        "draws":draws,"bootstrap_seed":seed,
        "training_eval_disjointness":"NOT established by evaluation equality; requires independent source ancestry audit",
        "interpretation":"Only use for real heldout reports sharing an immutable test protocol."
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--draws",type=int,default=10000)
    p.add_argument("--seed",type=int,default=314159)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("paired results already exist")
    report=summarize(a.eval_dir,draws=a.draws,seed=a.seed)
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report["contrasts"],indent=2))


if __name__=="__main__":main()
