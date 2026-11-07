#!/usr/bin/env python3
"""Collect six independently completed heldout cases; NEVER create results.

Each case must already have a verified original PBPF sandbox evaluation.
An explicit frozen source_component_id -> problem_id map is MANDATORY,
since upstream PBPF's task_id is not necessarily an independent problem.
The collected files are compatible with summarize_submitted_eed_token_factorial.py.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ARMS={
 "equal_uniform":"C01_EQUAL",
 "fixed_uniform":"C02_FIXED",
 "adaptive_uniform":"C03_EED",
 "fixed_edit":"C10_FIXED_EDIT",
 "adaptive_edit":"C11_EED_EDIT",
 "adaptive_random":"C12_EED_RANDOM_TOKEN",
}
IDENTITY=("model_id","revision","model_config_sha256",
          "shared_init_adapter_tree_sha256","response_token_budget",
          "seed","corrections_sha256","trainer_source_sha256")


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_problem_map(path:Path)->dict[str,str]:
    rows=[json.loads(x) for x in path.read_text().split("\n") if x.strip()]
    if not rows:raise ValueError("empty frozen source/problem map")
    mapping={}
    for r in rows:
        sid=r.get("source_component_id")
        pid=r.get("problem_id")
        if not isinstance(sid,str) or not sid or pid is None or str(pid)=="":
            raise ValueError("source map must give real source_component_id and problem_id")
        if sid in mapping:raise ValueError("duplicate source mapping")
        mapping[sid]=str(pid)
    return mapping


def collect(root:Path,study:str,seed:int,problem_map:Path,output:Path)->dict:
    if output.exists():raise FileExistsError("collection already exists")
    mapping=source_problem_map(problem_map)
    source_map_sha=sha(problem_map)
    cases={}
    first_setup=None
    first_eval=None
    raw={}
    for alias,case_id in ARMS.items():
        folder=root/study/case_id/f"seed{seed}"
        receipt=folder/"case.json"
        ep=folder/"evaluation"/"plan.json"
        result=folder/"evaluation"/"heldout"/"report.json"
        if not all(p.is_file() for p in (receipt,ep,result)):
            raise FileNotFoundError(f"{case_id}: case/evaluator receipt missing")
        case=json.loads(receipt.read_text())
        ev=json.loads(ep.read_text())
        report=json.loads(result.read_text())
        if (case.get("schema")!="eesd-iclr2027-independent-case-v1"
            or case.get("status")!="trained_awaiting_independent_evaluation"
            or case["case"]["id"]!=case_id
            or ev.get("status")!="finished_with_real_heldout"
            or ev.get("heldout_report_sha256")!=sha(result)):
            raise ValueError(f"{case_id}: not a fully certified real experiment")
        setup=case["setup"]
        if first_setup is None:first_setup=setup
        for field in IDENTITY:
            if setup.get(field)!=first_setup.get(field):
                raise ValueError(f"{case_id}: training settings differ in {field}")
        if ev["adapter_tree_sha256"]!=case["trained_adapter_tree_sha256"]:
            raise ValueError(f"{case_id}: evaluated adapter is not the trained one")
        identity={k:report.get(k) for k in (
           "model","revision","task_manifest_sha256","execution_profile",
           "execution_lock_sha256","sources")}
        if first_eval is None:first_eval=identity
        if identity!=first_eval:
            raise ValueError(f"{case_id}: evaluation protocol or population differs")
        rows=report.get("records")
        if not isinstance(rows,list) or not rows:
            raise ValueError(f"{case_id}: no raw evaluator records")
        source_ids=[z.get("source_component_id") for z in rows]
        if len(set(source_ids))!=len(source_ids) or set(source_ids)!=set(mapping):
            raise ValueError(f"{case_id}: missing/extra independent problem mapping")
        if any(not isinstance(z.get("all_tests_pass"),bool) for z in rows):
            raise ValueError(f"{case_id}: nonboolean all-tests success")
        for z in rows:
            if "problem_id" in z and str(z["problem_id"])!=mapping[z["source_component_id"]]:
                raise ValueError("frozen problem mapping contradicts evaluator row")
            z["problem_id"]=mapping[z["source_component_id"]]
        cases[alias]={
            "case_id":case_id,"case_sha256":sha(receipt),
            "evaluation_plan_sha256":sha(ep),
            "raw_evaluation_report_sha256":sha(result),
            "adapter_tree_sha256":ev["adapter_tree_sha256"],
        }
        raw[alias]=report
    output.mkdir(parents=True,exist_ok=False)
    for alias,value in raw.items():
        with (output/(alias+".json")).open("x") as f:
            json.dump(value,f,sort_keys=True,indent=2)
            f.write("\n")
        cases[alias]["collected_sha256"]=sha(output/(alias+".json"))
    manifest={
      "schema":"eesd-iclr2027-six-arm-heldout-collection-v1",
      "status":"six_real_heldout_reports_verified_not_statistically_interpreted",
      "study":study,"seed":seed,
      "problem_map_sha256":source_map_sha,
      "source_count":len(mapping),
      "distinct_problem_count":len(set(mapping.values())),
      "evaluation_identity":first_eval,
      "common_training_parameters":{k:first_setup[k] for k in IDENTITY},
      "arms":cases,
      "limitation":"This check verifies source/problem IDs against a user-supplied map, but does not independently authenticate that the map originated from the official dataset. Preserve its creation receipt and cluster by problem_id.",
    }
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases-root",type=Path,default=Path("runs/eesd-iclr2027-rebuttal"))
    p.add_argument("--study",required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--problem-map",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=collect(a.cases_root,a.study,a.seed,a.problem_map,a.output)
    print(json.dumps({"status":result["status"],"study":a.study,
                      "sources":result["source_count"],
                      "problems":result["distinct_problem_count"]},indent=2))


if __name__=="__main__":main()
