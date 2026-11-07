"""Tests for independently prepared, immutable reviewer experiment cases.

These tests do NOT launch a GPU, execute untrusted code, or invent results.
"""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/"scripts"
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location("rebuttal_cases",SCRIPTS/"run_eesd_rebuttal_case.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Args:
    def __init__(self,corrections,config,init,eval_ids=None,*,seed=1701,study="synthetic_test"):
        self.corrections=corrections
        self.model_config=config
        self.init_adapter=init
        self.eval_ids=eval_ids
        self.seed=seed
        self.study=study
        self.budget=256
        self.prediction_rows=None
        self.require_problem_disjoint=False


def row(tid,split,problem,rel=None,original=True):
    r={
        "trajectory_id":tid,"source_component_id":tid,
        "problem_id":problem,"split":split,
        "prompt":"Fix the following bug:\n"+json.dumps({"program":"a=1\nprint(a)"}),
        "correction":"a=2\nprint(a)","before_outcomes":[1,1,1,1],
        "after_outcomes":[0,0,1,0],
        "relevance":rel or [1.,1.,1.,1.]
    }
    if original:r["original"]="a=1\nprint(a)"
    return r


def fixture_paths(tmp_path,original=True):
    input_file=tmp_path/"corrections.jsonl"
    rows=[row("train1","train","P1"),row("train2","train","P2",rel=[5.,1.,1.,1.]),
          row("development","development","P3")]
    if not original:
        for r in rows:
            r.pop("original")
            r["prompt"]="Solve programming task with no source code."
    input_file.write_text("".join(json.dumps(r)+"\n" for r in rows))
    adapter=tmp_path/"shared_init"
    adapter.mkdir()
    (adapter/"adapter_config.json").write_text('{"base_model_name_or_path":"synthetic"}')
    config=ROOT/"configs/rebuttal/qwen25_coder_7b.yaml"
    evaluation=tmp_path/"evaluation_ids.jsonl"
    evaluation.write_text(json.dumps({
        "source_component_id":"test1","problem_id":"P4"
    })+"\n")
    return Args(input_file,config,adapter,evaluation)


def test_full_registered_case_count_and_mapping():
    cases=m.load_cases()
    assert len(cases)==21
    assert len([x for x in cases.values() if x["kind"]=="train"])==16
    assert all(x["reviewers"] for x in cases.values())
    assert cases["C03_EED"]["weight"]=="adaptive_eed"
    assert cases["C11_EED_EDIT"]["token"]=="edit"
    assert cases["C14_KAPPA32"]["kappa"]==32


def test_case_preparation_is_independent_sealed_and_never_overwritten(tmp_path):
    a=fixture_paths(tmp_path)
    cases=m.load_cases()
    p1=tmp_path/"outputs"/"a"/"C01_EQUAL"/"seed1701"
    p2=tmp_path/"outputs"/"a"/"C03_EED"/"seed1701"
    x=m.prepare_case(a,cases["C01_EQUAL"],p1)
    y=m.prepare_case(a,cases["C03_EED"],p2)
    assert x["status"]==y["status"]=="prepared"
    assert not (p1/"model").exists()
    assert len(list((tmp_path/"outputs").rglob("scored-corrections.jsonl")))==2
    weights1=[json.loads(r)["training_weights"]["eesd_full"]
              for r in (p1/"scored-corrections.jsonl").read_text().split("\n") if r]
    weights2=[json.loads(r)["training_weights"]["eesd_full"]
              for r in (p2/"scored-corrections.jsonl").read_text().split("\n") if r]
    assert all(v==1. for v in weights1)
    assert any(v!=1. for v in weights2)
    assert x["setup"]["shared_init_adapter_tree_sha256"]==y["setup"]["shared_init_adapter_tree_sha256"]
    assert "--include-zero-weight-kl" in x["command"]
    assert "--anchor-beta" in x["command"]
    assert "--token-weighting" in x["command"]
    with pytest.raises(FileExistsError):
        m.prepare_case(a,cases["C01_EQUAL"],p1)
    with pytest.raises(ValueError,match="checksum changed"):
        (a.init_adapter/"adapter_config.json").write_text("edited")
        m.check_source_integrity(x["setup"])


def test_edit_cases_reject_synthesis_records_even_when_uniform_allowed(tmp_path):
    a=fixture_paths(tmp_path,original=False)
    baseline=tmp_path/"uniform"
    m.prepare_case(a,m.load_cases()["C03_EED"],baseline)
    with pytest.raises(ValueError,match="ineligible"):
        m.prepare_case(a,m.load_cases()["C11_EED_EDIT"],tmp_path/"edit")


def test_kappa_cases_are_distinct_and_weight_diagnostics_are_saved(tmp_path):
    a=fixture_paths(tmp_path)
    cases=m.load_cases()
    original=m.prepare_case(a,cases["C03_EED"],tmp_path/"K16")
    altered=m.prepare_case(a,cases["C13_KAPPA8"],tmp_path/"K8")
    assert original["setup"]["alpha"]==altered["setup"]["alpha"]==.1
    a16=json.loads((tmp_path/"K16"/"weight_diagnostics.json").read_text())
    a8=json.loads((tmp_path/"K8"/"weight_diagnostics.json").read_text())
    assert a16["kappa"]==16 and a8["kappa"]==8
    assert original["frozen_case_input_hashes"]!=altered["frozen_case_input_hashes"]


def test_strict_source_disjointness_rejects_missing_problem_ids(tmp_path):
    a=fixture_paths(tmp_path)
    rows=[json.loads(s) for s in a.corrections.read_text().split("\n") if s]
    for r in rows:r.pop("problem_id")
    a.corrections.write_text("".join(json.dumps(r)+"\n" for r in rows))
    a.require_problem_disjoint=True
    with pytest.raises(ValueError,match="strict source-independent"):
        m.prepare_case(a,m.load_cases()["C01_EQUAL"],tmp_path/"no_problem_ids")




def test_lexical_relevance_uniform_and_permuted_keep_original_data_untouched(tmp_path):
    a=fixture_paths(tmp_path)
    cases=m.load_cases()
    original_data=a.corrections.read_bytes()
    base=m.prepare_case(a,cases["C03_EED"],tmp_path/"orig")
    uni=m.prepare_case(a,cases["C15_UNIFORM_RELEVANCE"],tmp_path/"uniform")
    perm=m.prepare_case(a,cases["C16_PERMUTED_RELEVANCE"],tmp_path/"permuted")
    assert a.corrections.read_bytes()==original_data
    def rows(folder):
        return [json.loads(x) for x in
            (tmp_path/folder/"scored-corrections.jsonl").read_text().split("\n") if x]
    b,u,p=rows("orig"),rows("uniform"),rows("permuted")
    assert b[1]["relevance"]!=u[1]["relevance"]
    assert sorted(b[1]["relevance"])==sorted(p[1]["relevance"])
    assert all(z["relevance"]==[1.,1.,1.,1.] for z in u)
    diag=json.loads((tmp_path/"permuted"/"weight_diagnostics.json").read_text())
    assert diag["relevance_mode"]=="permuted"
    assert base["status"]==uni["status"]==perm["status"]=="prepared"


def test_diagnostic_is_per_case_and_never_performs_gpu_update(tmp_path):
    a=fixture_paths(tmp_path)
    case=m.load_cases()["D02_RELEVANCE_DIAGNOSTIC"]
    out=tmp_path/"D02"
    data=m.prepare_case(a,case,out)
    assert (out/"relevance_diagnostic.json").is_file()
    assert data["command"] is None
    result=m.run_case(out)
    assert result["status"]=="completed_without_gpu"
    with pytest.raises(ValueError,match="not in prepared state"):
        m.run_case(out)
