import importlib.util
import json
from pathlib import Path
import pytest

S=Path(__file__).resolve().parents[1]/"scripts"/"audit_eesd_token_aware_inputs.py"
spec=importlib.util.spec_from_file_location("audit_tokenaware",S)
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def _rec(tid,split,source,program=None,problem=None):
    row={"trajectory_id":tid,"source_component_id":source,"split":split,
         "correction":"def f(x): return x+1",
         "prompt":"Solve Python:\n"+json.dumps({"program":"def f(x): return x"})}
    if program is not None:row["original"]=program
    if problem is not None:row["problem_id"]=problem
    return row


def _write(path,rows):
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    return path


def test_audit_reports_edit_eligibility_and_missing_origins(tmp_path):
    path=_write(tmp_path/"corrections.jsonl",[
        _rec("a","train","sourceA",problem="pA"),
        {"trajectory_id":"b","source_component_id":"sourceB",
         "split":"development","correction":"print(4)",
         "prompt":"Write a program for integers."}
    ])
    r=audit.audit(path)
    assert r["row_count"]==2
    assert r["recoverable_original_count"]==1
    assert r["can_run_complete_edit_ablation"] is False
    assert r["missing_original_count"]==1
    assert r["nonidentical_correction_pairs"]==1


def test_audit_rejects_development_and_test_overlap(tmp_path):
    path=_write(tmp_path/"corr.jsonl",[
        _rec("a","train","A","def f(x): return x",problem="A"),
        _rec("b","development","B","def f(x): return x",problem="B"),
    ])
    evalfile=_write(tmp_path/"eval.jsonl",[{"source_component_id":"Q","problem_id":"A"}])
    with pytest.raises(ValueError,match="evaluation problem IDs overlap"):
        audit.audit(path,eval_ids=evalfile)
    ev2=_write(tmp_path/"eval2.jsonl",[{"source_component_id":"Q","problem_id":"Q"}])
    r=audit.audit(path,eval_ids=ev2)
    assert r["evaluation"]["problem_level_overlap_checked"]


def test_audit_rejects_conflicting_originals(tmp_path):
    path=_write(tmp_path/"corr.jsonl",[
        _rec("a","train","A",program="not the prompt program"),
    ])
    res=audit.audit(path)
    assert res["can_run_complete_edit_ablation"] is False
