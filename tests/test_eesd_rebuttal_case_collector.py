"""Synthetic-only six-case receipt collection: not scientific outcomes."""
import importlib.util
import json
from pathlib import Path

import pytest

S=Path(__file__).resolve().parents[1]/"scripts"/"collect_eesd_rebuttal_case_evals.py"
spec=importlib.util.spec_from_file_location("independent_collector",S)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def _fixture(tmp_path):
    study="unit_test"
    model="Qwen/Qwen2.5-Coder-7B-Instruct"
    setup={
      "model_id":model,"revision":"a"*40,"model_config_sha256":"b"*64,
      "shared_init_adapter_tree_sha256":"c"*64,"response_token_budget":512,
      "seed":1701,"corrections_sha256":"d"*64,"trainer_source_sha256":"e"*64
    }
    path=tmp_path/"map.jsonl"
    path.write_text(
        json.dumps({"source_component_id":"s1","problem_id":"p1"})+"\n"+
        json.dumps({"source_component_id":"s2","problem_id":"p2"})+"\n"
    )
    for i,(alias,case_id) in enumerate(m.ARMS.items()):
        folder=tmp_path/study/case_id/"seed1701"
        heldout=folder/"evaluation"/"heldout"
        heldout.mkdir(parents=True)
        report={
            "schema":"eesd-fresh-policy-eval-v1","model":model,"revision":"a"*40,
            "task_manifest_sha256":"f"*64,"execution_profile":"sandbox",
            "execution_lock_sha256":None,"sources":2,
            "records":[
                {"source_component_id":"s1","task_id":"task1","all_tests_pass":True},
                {"source_component_id":"s2","task_id":"task2",
                 "all_tests_pass":bool(i%2)}
            ]
        }
        p=heldout/"report.json"
        p.write_text(json.dumps(report)+"\n")
        (folder/"case.json").write_text(json.dumps({
            "schema":"eesd-iclr2027-independent-case-v1",
            "status":"trained_awaiting_independent_evaluation",
            "case":{"id":case_id,"kind":"train"},
            "setup":setup,
            "trained_adapter_tree_sha256":f"adapter-{i}"
        })+"\n")
        (folder/"evaluation"/"plan.json").write_text(json.dumps({
            "status":"finished_with_real_heldout",
            "adapter_tree_sha256":f"adapter-{i}",
            "heldout_report_sha256":m.sha(p)
        })+"\n")
    return study,path


def test_collect_six_cases_isolates_origins_and_real_problem_map(tmp_path):
    study,path=_fixture(tmp_path)
    output=tmp_path/"out"
    report=m.collect(tmp_path,study,1701,path,output)
    assert report["source_count"]==2
    assert report["distinct_problem_count"]==2
    assert set(report["arms"])==set(m.ARMS)
    r=json.loads((output/"adaptive_edit.json").read_text())
    assert r["records"][0]["problem_id"]=="p1"
    assert r["records"][1]["problem_id"]=="p2"
    with pytest.raises(FileExistsError):
        m.collect(tmp_path,study,1701,path,output)


def test_collector_rejects_mismatched_test_manifest(tmp_path):
    study,path=_fixture(tmp_path)
    p=tmp_path/study/"C12_EED_RANDOM_TOKEN"/"seed1701"/"evaluation"/"heldout"/"report.json"
    r=json.loads(p.read_text());r["task_manifest_sha256"]="wrong";p.write_text(json.dumps(r))
    plan=p.parent.parent/"plan.json"
    d=json.loads(plan.read_text())
    d["heldout_report_sha256"]=m.sha(p)
    plan.write_text(json.dumps(d)+"\n")
    with pytest.raises(ValueError,match="evaluation protocol"):
        m.collect(tmp_path,study,1701,path,tmp_path/"out")
