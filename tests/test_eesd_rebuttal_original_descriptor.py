"""Original 4-state EED descriptor/partial repair tests, no GPU/results."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

S=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(S))
spec=importlib.util.spec_from_file_location("original_descriptor_audit",S/"audit_eesd_original_descriptor.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fake(before,after,*,tid,split="train"):
    descriptors=[{"input":str(i),"expected":"2","actual":"1","stderr":"",
                  "outcome":"PASS" if code==0 else "WRONG_OUTPUT"}
                 for i,code in enumerate(before)]
    return {
        "source_component_id":tid,"trajectory_id":tid,"split":split,
        "original":"print(1)",
        "correction":"print(2)",
        "prompt":"Repair program\n"+json.dumps({
            "program":"print(1)","public_executions":descriptors
        }),
        "relevance":[1.,1.,1.,1.],
        "before_outcomes":before,
        "after_outcomes":after,
        "relevance_kernel":{
            "type":"frozen_signed_hash_cosine",
            "dimension":256,"strength":16,
            "query":"unified_diff(original, correction)",
            "history":"public execution descriptor only"
        }
    }


def test_reports_original_descriptor_ancestry_and_partial_success_not_global_pass(tmp_path):
    rows=[fake([1,1,1,1],[0,1,1,1],tid="one"),
          fake([1,0,0,0],[0,0,0,0],tid="two",split="development")]
    f=tmp_path/"orig.jsonl"
    f.write_text("".join(json.dumps(x)+"\n" for x in rows))
    r=m.audit(f)
    assert r["rows"]==2
    assert r["kernel_metadata"]["original_metadata_verified"]==2
    assert r["descriptor_coverage"]["complete_five_field_public_descriptor"]==2
    assert r["descriptor_coverage"]["before_outcomes_consistent_with_descriptor"]==2
    assert r["per_split"]["train"]["positive_weight_but_not_fully_correct"]==1
    assert r["jtox_counterexample"]["weights_equal"]
    assert r["jtox_counterexample"]["after_public_pass_rate_unresolved"]==.25
    assert r["full_kernel_recomputation_attempted"] is False


def test_partial_descriptor_is_not_claimed_reconstructed(tmp_path):
    row=fake([1,1,1,1],[0,1,1,1],tid="a")
    obj=json.loads(row["prompt"].split("\n",1)[1])
    obj["public_executions"][0].pop("actual")
    row["prompt"]="Repair program\n"+json.dumps(obj)
    p=tmp_path/"f.jsonl";p.write_text(json.dumps(row)+"\n")
    r=m.audit(p)
    assert r["descriptor_coverage"]["missing_or_truncated_public_descriptor"]==1
    assert "recomputed_relevance_matches_original" not in r["descriptor_coverage"]
