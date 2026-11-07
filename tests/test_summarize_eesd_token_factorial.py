"""Synthetic-only pairing tests; NO scientific held-out accuracy is invented."""
import importlib.util
import json
from pathlib import Path

import pytest

P=Path(__file__).resolve().parents[1]/"scripts"/"summarize_submitted_eed_token_factorial.py"
spec=importlib.util.spec_from_file_location("summarize_tokenfactorial",P)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def _report(y):
    return {
        "schema":"eesd-fresh-policy-eval-v1",
        "domain":"codearc","model":"some-backbone",
        "revision":"a"*40,"task_manifest_sha256":"x"*64,
        "execution_lock_sha256":"y"*64,"execution_profile":"sandbox",
        "records":[{"source_component_id":f"source-{i}",
                    "problem_id":f"problem-{i}",
                    "all_tests_pass":bool(z)}
                   for i,z in enumerate(y)]
    }


def _make(tmp_path):
    outcomes={
        "equal_uniform":[0,0,1,0],
        "fixed_uniform":[0,1,1,0],
        "adaptive_uniform":[1,1,1,0],
        "fixed_edit":[1,1,0,0],
        "adaptive_edit":[1,1,1,1],
        "adaptive_random":[0,1,1,1]
    }
    for arm,flags in outcomes.items():
        (tmp_path/(arm+".json")).write_text(json.dumps(_report(flags)))
    return outcomes


def test_2x2_factorial_contrasts_and_specificity(tmp_path):
    _make(tmp_path)
    out=m.summarize(tmp_path,draws=300,seed=17)
    assert out["source_count"]==4
    assert out["independent_problem_count"]==4
    assert out["contrasts"]["adaptive_mass_uniform"]["problem_mean_effect"]==pytest.approx(.25)
    assert out["contrasts"]["adaptive_edit_gain"]["problem_mean_effect"]==pytest.approx(.25)
    assert out["contrasts"]["edit_mass_interaction"]["problem_mean_effect"]==pytest.approx(.25)
    assert out["contrasts"]["edited_vs_random_same_size"]["problem_mean_effect"]==pytest.approx(.25)
    assert out["paired_fix_regress"]["adaptive_edit__vs__adaptive_uniform"]["fixes"]==1
    assert out["paired_fix_regress"]["adaptive_edit__vs__adaptive_uniform"]["regressions"]==0


def test_fails_if_evaluation_population_or_lock_changes(tmp_path):
    _make(tmp_path)
    path=tmp_path/"adaptive_random.json"
    data=json.loads(path.read_text())
    data["execution_lock_sha256"]="tampered"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="mismatched evaluator"):
        m.summarize(tmp_path,draws=300)


def test_no_problem_identifier_means_no_independence_claim(tmp_path):
    _make(tmp_path)
    path=tmp_path/"adaptive_edit.json"
    data=json.loads(path.read_text())
    for row in data["records"]:row.pop("problem_id")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="problem ID changed"):
        m.summarize(tmp_path,draws=300)
