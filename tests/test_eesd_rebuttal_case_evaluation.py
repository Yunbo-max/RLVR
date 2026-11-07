"""Synthetic-only original PBPF evaluation preparation tests; no GPU or code execution."""
import importlib.util
import json
from pathlib import Path

import pytest

from eesd.training_outcomes import tree_sha

S=Path(__file__).resolve().parents[1]/"scripts"/"run_eesd_rebuttal_case_evaluation.py"
spec=importlib.util.spec_from_file_location("single_case_evaluator",S)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _case(tmp):
    case=tmp/"case"
    adapter=case/"model"/"adapter"
    adapter.mkdir(parents=True)
    (adapter/"adapter_config.json").write_text('{"r": 16}')
    evalfile=tmp/"eval_sources.jsonl"
    evalfile.write_text(json.dumps({"source_component_id":"source_a","problem_id":"p1"})+"\n")
    config=tmp/"model.yaml"
    config.write_text("model_id: Qwen/Qwen2.5-Coder-7B-Instruct\nrevision: c03e6d358207e414f1eca0bb1891e29f1db0e242\n")
    setup={
        "model_id":"Qwen/Qwen2.5-Coder-7B-Instruct",
        "revision":"c03e6d358207e414f1eca0bb1891e29f1db0e242",
        "model_config_path":str(config),
        "model_config_sha256":mod.file_sha(config),
    }
    receipt={
        "schema":"eesd-iclr2027-independent-case-v1",
        "status":"trained_awaiting_independent_evaluation",
        "case":{"kind":"train","id":"C03_EED"},
        "study":"qwen_rbr","seed":1701,
        "original_paper_cell":"runbugrun__qwen25_7b",
        "setup":setup,
        "trained_adapter_tree_sha256":tree_sha(adapter),
        "frozen_case_input_hashes":{"eval_ids_sha256":mod.file_sha(evalfile)},
    }
    (case/"case.json").write_text(json.dumps(receipt)+"\n")
    (case/"data_provenance.json").write_text(json.dumps({
        "provenance_status":"problem_ids_disjoint_checked"})+"\n")
    receipt["data_provenance_sha256"]=mod.file_sha(case/"data_provenance.json")
    (case/"case.json").write_text(json.dumps(receipt)+"\n")
    pbpf=tmp/"pbpf";pbpf.mkdir()
    public=tmp/"public";public.mkdir()
    evaluator=tmp/"evaluator";evaluator.mkdir()
    return case,pbpf,public,evaluator,evalfile


def test_prepared_case_requires_identical_frozen_evaluation_ids(tmp_path,monkeypatch):
    case,pbpf,public,evaluator,ids=_case(tmp_path)
    monkeypatch.setattr(mod,"validate_checkout",lambda *_:{
        "pbpf_commit":mod.PBPF_PIN,
        "generator_script":"pinned-original-generator",
        "generator_sha256":"a"*64,
        "evaluator_script":"pinned-original-evaluator",
        "evaluator_sha256":"b"*64})
    monkeypatch.setattr(mod,"inventory_ids",lambda *_:{
        "public_manifest_sha256":"a"*64,
        "public_tasks_sha256":"b"*64,
        "evaluator_manifest_sha256":"c"*64,
        "evaluator_tasks_sha256":"d"*64,
        "evaluation_source_ids":["source_a"],"evaluation_sources":1})
    with pytest.raises(ValueError,match="evaluation source-ID file"):
        mod.prepare(case,pbpf,public,evaluator,"rbr",12)
    plan=mod.prepare(case,pbpf,public,evaluator,"rbr",12,ids)
    assert plan["status"]=="prepared"
    assert plan["source_disjointness_status"]=="problem_ids_disjoint_checked"
    assert plan["eval_source_id_file_sha256"]==mod.file_sha(ids)
    assert plan["generation_command"].count("--greedy")==1
    assert plan["heldout_command"][-3:]==["sandbox","--timeout","6.0"]
    assert plan["adapter_tree_sha256"]==tree_sha(case/"model"/"adapter")
    with pytest.raises(FileExistsError):
        mod.prepare(case,pbpf,public,evaluator,"rbr",12,ids)


def test_malformed_evaluator_id_bank_is_rejected(tmp_path,monkeypatch):
    case,pbpf,public,evaluator,ids=_case(tmp_path)
    monkeypatch.setattr(mod,"validate_checkout",lambda *_:{
        "pbpf_commit":mod.PBPF_PIN,
        "generator_script":"pinned","generator_sha256":"a"*64,
        "evaluator_script":"pinned","evaluator_sha256":"b"*64})
    monkeypatch.setattr(mod,"inventory_ids",lambda *_:{
        "public_manifest_sha256":"a"*64,
        "public_tasks_sha256":"b"*64,
        "evaluator_manifest_sha256":"c"*64,
        "evaluator_tasks_sha256":"d"*64,
        "evaluation_source_ids":["different"],"evaluation_sources":1})
    with pytest.raises(ValueError,match="do not match actual"):
        mod.prepare(case,pbpf,public,evaluator,"rbr",12,ids)


def test_legacy_bridge_rejects_cross_domain_case_before_evaluation(tmp_path, monkeypatch):
    case,pbpf,public,evaluator,ids=_case(tmp_path)
    monkeypatch.setattr(mod,"validate_checkout",lambda *_:{})
    monkeypatch.setattr(mod,"inventory_ids",lambda *_:{"evaluation_source_ids":["source_a"]})
    with pytest.raises(ValueError,match="cell.*domain"):
        mod.prepare(case,pbpf,public,evaluator,"codearc",12,ids)
    assert not (case/"evaluation").exists()


def test_legacy_bridge_rejects_changed_provenance_or_model_yaml(tmp_path, monkeypatch):
    case,pbpf,public,evaluator,ids=_case(tmp_path)
    monkeypatch.setattr(mod,"validate_checkout",lambda *_:{})
    monkeypatch.setattr(mod,"inventory_ids",lambda *_:{"evaluation_source_ids":["source_a"]})
    provenance=case/"data_provenance.json"
    original=provenance.read_text()
    provenance.write_text('{"provenance_status":"edited"}')
    with pytest.raises(ValueError,match="provenance"):
        mod.prepare(case,pbpf,public,evaluator,"rbr",12,ids)
    provenance.write_text(original)
    (tmp_path/"model.yaml").write_text("model_id: altered\nrevision: " + "a" * 40)
    with pytest.raises(ValueError,match="model config"):
        mod.prepare(case,pbpf,public,evaluator,"rbr",12,ids)
