"""CI checks: 17 truly separate Python entrypoints, 4 audits, all 3x4 cells.

No GPU or untrusted candidate program is executed in these tests.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from rebuttal_test_utils import write_lora_adapter

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/"scripts"
CASES=SCRIPTS/"rebuttal_cases"
spec=importlib.util.spec_from_file_location("eesd_coverage",SCRIPTS/"check_eesd_rebuttal_coverage.py")
coverage=importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


def test_model_dataset_matrix_matches_exact_original_twelve_results():
    m=coverage.load_matrix()
    assert len(m)==12
    assert set(m)=={f"{d}__{model}" for d in coverage.DATASETS
                    for model in coverage.MODELS}
    assert sum(x["primary_evaluator"]=="pinned_pbpf_sandbox" for x in m.values())==4
    assert sum(x["primary_evaluator"]!="pinned_pbpf_sandbox" for x in m.values())==8
    assert all(x["matched_evaluator"]=="prospective_matched_greedy_sandbox" for x in m.values())
    assert m["codearc__deepseek_6p7b"]["original_paper"]["original_eed_solved"]==102
    assert m["apps_replay__gemma3_4b"]["original_paper"]["original_eed_solved"]==2


def test_17_case_python_files_are_one_to_one_not_one_batch_script():
    reg=json.loads((ROOT/"configs/rebuttal/eesd_case_registry.json").read_text())
    all_cases={entry["id"] for entry in reg["cases"]}
    c={key for key in all_cases if key.startswith("C")}
    d={key for key in all_cases if key.startswith("D")}
    assert len(c)==17 and len(d)==4
    assert {int(k[1:3]) for k in c}==set(range(17))
    scripts=[p for p in CASES.glob("*.py") if p.name!="_entry.py"]
    assert len(scripts)==21
    launched=[]
    for file in scripts:
        node=ast.parse(file.read_text())
        invocations=[
            n for n in ast.walk(node) if isinstance(n,ast.Call)
            and isinstance(n.func,ast.Name) and n.func.id=="main_for"
        ]
        assert len(invocations)==1,file
        args=invocations[0].args
        assert len(args)==1 and isinstance(args[0],ast.Constant),file
        launched.append(args[0].value)
    assert set(launched)==all_cases and len(launched)==len(set(launched))


def test_all_seventeen_launcher_help_commands_work_without_data():
    # Smoke the actual real Python entrypoint, not only its AST wrapper.
    for p in sorted(CASES.glob("c[0-1][0-9]_*.py")):
        r=subprocess.run([sys.executable,str(p),"--help"],cwd=ROOT,
                         text=True,capture_output=True,timeout=15)
        assert r.returncode==0,(p,r.stderr)
        assert "--cell" in r.stdout
        assert "--action" in r.stdout


def test_named_cell_rejects_backbone_mismatch_before_training(tmp_path):
    file=tmp_path/"input.jsonl"
    record={
        "split":"train","trajectory_id":"t0","source_component_id":"s0",
        "model_id":"Qwen/Qwen2.5-Coder-7B-Instruct",
        "model_revision":coverage.load_matrix()["codearc__qwen25_7b"]["model_revision"],
        "domain":"codearc","prompt":"repair","correction":"print(42)",
        "before_outcomes":[1],"after_outcomes":[0],"relevance":[1.]
    }
    dev={**record,"split":"development","trajectory_id":"t1","source_component_id":"s1"}
    file.write_text(json.dumps(record)+"\n"+json.dumps(dev)+"\n")
    valid=coverage.audit_correction_identity("codearc__qwen25_7b",file,strict=True)
    assert valid["status"]=="verified_model_and_dataset_identity"
    assert valid["rows"]==2
    with pytest.raises(ValueError,match="correction model_id"):
        coverage.audit_correction_identity("codearc__deepseek_6p7b",file)
    with pytest.raises(ValueError,match="correction domain"):
        coverage.audit_correction_identity("runbugrun__qwen25_7b",file)


def test_model_dataset_coverage_discloses_missing_inputs_and_unverified_eval():
    rows=coverage.coverage_rows()
    assert len(rows)==12
    assert all(x["correction_data"]=="not_supplied" for x in rows)
    assert all(x["evaluation_claim_allowed"] is False for x in rows)
    assert all(x["training_code"] for x in rows)
    assert sum(x["evaluation_code"]=="pinned_pbpf_sandbox" for x in rows)==4
    assert all(x["matched_evaluation_code"]=="prospective_matched_greedy_sandbox" for x in rows)
    assert all("not_verified" in x["matched_execution_status"] for x in rows)


def test_standalone_launcher_requires_cell_and_fixes_case_id(tmp_path):
    script=CASES/"c03_eed.py"
    bad=subprocess.run([sys.executable,str(script),"--action","prepare",
                        "--out-root",str(tmp_path)],cwd=ROOT,
                       text=True,capture_output=True,timeout=15)
    assert bad.returncode!=0
    assert "require --cell" in bad.stderr

    mismatch=subprocess.run([
        sys.executable,str(script),"--case","C02_FIXED",
        "--cell","runbugrun__qwen25_7b","--action","prepare",
        "--out-root",str(tmp_path)],cwd=ROOT,
        text=True,capture_output=True,timeout=15)
    assert mismatch.returncode!=0
    assert "cannot run another --case" in mismatch.stderr



def test_two_real_python_launchers_prepare_distinct_cases_without_gpu(tmp_path):
    cell="codearc__qwen25_7b"
    cfg=ROOT/"configs/rebuttal/qwen25_coder_7b.yaml"
    revision=coverage.load_matrix()[cell]["model_revision"]
    raw=[]
    for i,split in enumerate(("train","development")):
        raw.append({
            "trajectory_id":f"traj{i}","source_component_id":f"src{i}",
            "split":split,"problem_id":f"prob{i}","domain":"codearc",
            "model_id":"Qwen/Qwen2.5-Coder-7B-Instruct",
            "model_revision":revision,
            "prompt":"Repair source\n"+json.dumps({"program":"print(0)"}),
            "original":"print(0)","correction":"print(1)",
            "before_outcomes":[1,1,1,1],"after_outcomes":[0,0,1,0],
            "relevance":[.25,.25,.25,.25],
        })
    data=tmp_path/"input.jsonl"
    data.write_text("".join(json.dumps(x)+"\n" for x in raw))
    initial=tmp_path/"init"
    write_lora_adapter(initial, cfg)
    out=tmp_path/"runs"

    def invoke(file):
        return subprocess.run([
            sys.executable,str(CASES/file),
            "--cell",cell,
            "--model-config",str(cfg),
            "--corrections",str(data),
            "--init-adapter",str(initial),
            "--seed","1701","--budget","128",
            "--out-root",str(out),
            "--strict-record-identity",
            "--action","prepare"
        ],cwd=ROOT,text=True,capture_output=True,timeout=20)

    first=invoke("c02_fixed.py")
    second=invoke("c03_eed.py")
    assert first.returncode==0,first.stderr
    assert second.returncode==0,second.stderr
    a=out/cell/"C02_FIXED"/"seed1701"
    b=out/cell/"C03_EED"/"seed1701"
    assert (a/"scored-corrections.jsonl").is_file()
    assert (b/"scored-corrections.jsonl").is_file()
    assert json.loads((a/"case.json").read_text())["original_paper_cell"]==cell
    assert json.loads((b/"case.json").read_text())["cell_identity_audit"]["status"]=="verified_model_and_dataset_identity"
    assert json.loads((a/"case.json").read_text())["case"]["id"]=="C02_FIXED"
    assert json.loads((b/"case.json").read_text())["case"]["id"]=="C03_EED"
    assert (a/"scored-corrections.jsonl").read_bytes()!=(b/"scored-corrections.jsonl").read_bytes()
    repeated=invoke("c02_fixed.py")
    assert repeated.returncode!=0
    assert "already exists" in repeated.stderr
