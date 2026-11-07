#!/usr/bin/env python3
"""Fail-closed census before spending GPU time on reviewer-motivated token edits.

Read scored + original four-state EED weights; record whether the **actual
original program** can be recovered for every correction. This is necessary
because CodeARC and replay synthesis examples may have no coherent before/after
program pair; inferring a synthetic original is prohibited.

No GPU, tokenizer, training, or held-out labels are used.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from eesd.token_aware import original_program


def digest(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def audit(path:Path,*,eval_ids:Path|None=None)->dict:
    raw=[json.loads(s) for s in path.read_text().split("\n") if s.strip()]
    if not raw:raise ValueError("empty correction corpus")
    split_ids={"train":set(),"development":set()}
    split_problems={"train":set(),"development":set()}
    provenance=Counter()
    problems_missing=Counter()
    errors=Counter()
    n_diffs=0
    missing_original=[]
    for row in raw:
        split=row.get("split")
        if split not in split_ids:raise ValueError("source split must be train/development")
        sid=row.get("source_component_id")
        if not isinstance(sid,str) or not sid:
            raise ValueError("source_component_id required")
        split_ids[split].add(sid)
        pid=row.get("problem_id")
        if pid is not None:
            split_problems[split].add(str(pid))
        else:problems_missing[split]+=1
        if not isinstance(row.get("correction"),str) or not row["correction"].strip():
            raise ValueError("missing correction source")
        try:
            before,field=original_program(row)
            provenance[field]+=1
            n_diffs+=int(before!=row["correction"])
        except ValueError as e:
            errors[str(e).split(":")[0]]+=1
            missing_original.append(row.get("trajectory_id",sid))
    if split_ids["train"] & split_ids["development"]:
        raise ValueError("overlap between train and development source ids")
    if split_problems["train"] and split_problems["development"]:
        if split_problems["train"] & split_problems["development"]:
            raise ValueError("train/development problem ids overlap")
    eval_report=None
    if eval_ids:
        ev=[json.loads(s) for s in eval_ids.read_text().split("\n") if s.strip()]
        eid={str(r["source_component_id"]) for r in ev if "source_component_id" in r}
        ep={str(r["problem_id"]) for r in ev if "problem_id" in r}
        if eid and (eid & (split_ids["train"]|split_ids["development"])):
            raise ValueError("evaluation source IDs overlap with training/development")
        if ep and (ep & (split_problems["train"]|split_problems["development"])):
            raise ValueError("evaluation problem IDs overlap with training/development")
        eval_report={
            "eval_input_sha256":digest(eval_ids),
            "eval_rows":len(ev),"eval_source_ids_supplied":len(eid),
            "eval_problem_ids_supplied":len(ep),
            "source_level_overlap_checked":bool(eid),
            "problem_level_overlap_checked":bool(ep) and
                not any(problems_missing.values()),
        }
    return {
        "schema":"eesd-token-aware-input-readiness-v1",
        "correction_file_sha256":digest(path),
        "row_count":len(raw),
        "split_source_counts":{k:len(v) for k,v in split_ids.items()},
        "split_problem_counts":{k:len(v) for k,v in split_problems.items()},
        "missing_problem_id_counts":dict(problems_missing),
        "recoverable_original_count":sum(provenance.values()),
        "missing_original_count":len(missing_original),
        "first_missing_original_ids":missing_original[:20],
        "original_sources":dict(provenance),
        "nonidentical_correction_pairs":n_diffs,
        "errors":dict(errors),
        "can_run_complete_edit_ablation":not missing_original,
        "evaluation":eval_report,
        "interpretation":"NO evaluation labels were used; exact token edit positions require the model tokenizer."
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corrections",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--evaluation-ids",type=Path)
    p.add_argument("--require-complete",action="store_true")
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("audit output exists")
    report=audit(a.corrections,eval_ids=a.evaluation_ids)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("row_count","recoverable_original_count",
        "missing_original_count","can_run_complete_edit_ablation")},indent=2))
    if a.require_complete and not report["can_run_complete_edit_ablation"]:
        raise SystemExit("FAIL: real original code missing; do not launch edit-aware training")


if __name__=="__main__":main()
