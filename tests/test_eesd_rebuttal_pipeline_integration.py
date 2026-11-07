"""CPU-only writer/collector integration; synthetic fixtures are NOT evidence.

The optional pinned PBPF checkout supplies source identity only; set
EESD_PBPF_SOURCE or place it beside the repo as pbpf-original. No model is loaded
and no generated program or sandbox is executed. Case preparation, provenance,
bank sealing, report writing, collection, and summary use real code.
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from eesd import rebuttal_eval as ev

ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "C00_FROZEN"
STUDY = "cpu-pipeline-contract-fixture"
SEED = 1701


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def _prepare_real_baseline(tmp_path, pbpf):
    # Reuse only materializations and the CPU adapter from the schema fixture;
    # its hand-written case/provenance receipts are never used by this pipeline.
    fixtures = _load_module("_pipeline_materializations",
                            ROOT / "tests/test_eesd_rebuttal_matched_evaluation.py")
    banks = fixtures.fixture(tmp_path / "materializations", baseline=True)
    runner = _load_module("_pipeline_case_runner", ROOT / "scripts/run_eesd_rebuttal_case.py")
    model_id, revision = ev.MODELS["qwen25_7b"]
    cell = "runbugrun__qwen25_7b"
    corrections = tmp_path / "synthetic-corrections.jsonl"
    _write_jsonl(corrections, [
        {"trajectory_id": f"synthetic-{split}", "source_component_id": f"synthetic-{split}",
         "problem_id": f"synthetic-{split}-problem", "split": split,
         "model_id": model_id, "model_revision": revision, "domain": "rbr",
         "dataset": "runbugrun", "prompt": "CPU contract fixture: fix the program.",
         "original": "print(0)", "correction": "print(1)",
         "before_outcomes": [1, 1, 1, 1], "after_outcomes": [0, 0, 0, 0],
         "relevance": [1.0, 1.0, 1.0, 1.0]}
        for split in ("train", "development")
    ])
    problem_map = tmp_path / "synthetic-evaluation-map.jsonl"
    tasks = ev.read_jsonl(banks["public"] / "tasks.jsonl")
    _write_jsonl(problem_map, [
        {"source_component_id": task["source_component_id"], "task_id": task["task_id"],
         "problem_id": f"synthetic-heldout-problem-{index // 5}", "domain": "rbr",
         "dataset": "runbugrun", "cell": cell, "model_id": model_id}
        for index, task in enumerate(tasks)
    ])
    args = SimpleNamespace(
        corrections=corrections, model_config=ROOT / "configs/rebuttal/qwen25_coder_7b.yaml",
        init_adapter=banks["case"] / "initial", eval_ids=problem_map,
        seed=SEED, study=STUDY, budget=256, prediction_rows=None,
        require_problem_disjoint=True, cell=cell,
        cell_identity=runner.cell_coverage.audit_correction_identity(cell, corrections, strict=True),
    )
    cases_root = tmp_path / "cases"
    case = runner.safe_path(cases_root, STUDY, CASE_ID, SEED)
    prepared = runner.prepare_case(args, runner.load_cases()[CASE_ID], case)
    assert prepared["command"] is None
    completed = runner.run_case(case)
    assert completed["status"] == "completed_diagnostic_without_gpu"
    provenance = ev.read_json(case / "data_provenance.json")
    assert provenance["provenance_status"] == "problem_ids_disjoint_checked"
    assert provenance["evaluation"]["eval_problem_ids_supplied"] == 100
    return {"case": case, "pbpf": pbpf, "public": banks["public"],
            "evaluator": banks["evaluator"], "eval_ids": problem_map, "workers": 1}, cases_root


def test_cpu_synthetic_writer_to_collector_and_paired_summary(tmp_path, monkeypatch):
    """Exercise emitted receipts end to end, never produce scientific evidence."""
    pbpf = Path(os.environ.get("EESD_PBPF_SOURCE", ROOT.parent / "pbpf-original")).resolve()
    if not pbpf.is_dir():
        pytest.skip(f"optional CPU integration requires clean PBPF {ev.PBPF_PIN} at EESD_PBPF_SOURCE")
    args, cases_root = _prepare_real_baseline(tmp_path, pbpf)
    case = args["case"]
    plan = ev.prepare(**args)
    assert plan["problem_disjointness_rechecked"] is True
    assert plan["case_id"] == CASE_ID

    # These labels deliberately cannot be mistaken for actual library versions
    # or executed-program evidence. The fake services exist only in this test.
    versions = {name: "TEST_ONLY_NOT_LOADED" for name in
                ("torch", "transformers", "peft", "bitsandbytes", "python")}
    execution_identity = {
        "execution_policy": {"profile": "sandbox", "test_only_fake_backend": True},
        "bwrap_sha256": "0" * 64, "python_sha256": "1" * 64,
        "executor_sha256": "2" * 64, "shared_limits_sha256": "3" * 64,
        "kernel": "TEST_ONLY_NO_SANDBOX_EXECUTED",
    }
    execution_lock = ev._json_sha(execution_identity)
    probes, stages, scored_sources = [], [], []

    def fake_probe(source, domain, timeout=6.0):
        assert Path(source).resolve() == pbpf and domain == "rbr" and timeout == 6.0
        probes.append(domain)
        return {**execution_identity, "execution_lock_sha256": execution_lock, "probe_outcome": "PASS"}

    def fake_generation_and_real_evaluation(command, log_path):
        assert Path(command[command.index("--case-dir") + 1]) == case
        phase = command[command.index("--_phase") + 1]
        stages.append(phase)
        Path(log_path).write_text("CPU integration fixture only; no model or program execution.\n")
        if phase == "generate":
            current, _ = ev._phase_plan(case, public_only=True)
            candidates = []
            for task in current["primary_tasks"]:
                fake_prompt = "CPU-ONLY SYNTHETIC PROMPT: " + task["source_component_id"]
                candidates.append({**task,
                                   "completion": "# Test fixture only; never executed.\nprint(0)\n",
                                   "program": "# Test fixture only; never executed.\nprint(0)\n",
                                   "input_tokens": 1, "output_tokens": 1, "cutoff": False,
                                   "prompt_sha256": hashlib.sha256(fake_prompt.encode()).hexdigest()})
            ev.seal_bank(current,
                         {"generator_dependency_versions": versions,
                          "generation_runtime": {"test_only_fake_generator": True,
                                                 "scientific_evidence": False}},
                         candidates)
        elif phase == "evaluate":
            ev.evaluate(case)
        else:
            raise AssertionError(f"unexpected test dispatch phase: {phase}")

    def fake_candidate_backend(payload):
        current, private, candidate = payload
        assert current["workers"] == 1
        assert private["task_id"] == candidate["task_id"]
        assert private["source_component_id"] == candidate["source_component_id"]
        source = candidate["source_component_id"]
        scored_sources.append(source)
        all_pass = int(source.rsplit("-", 1)[1]) % 2 == 0
        tests = [{"test_id": test["id"], "hidden": index >= 4,
                  "outcome": "PASS" if all_pass or index < 4 else "WRONG_OUTPUT",
                  "test_only_fake_execution": True}
                 for index, test in enumerate(private["tests"])]
        return {"task_id": candidate["task_id"], "source_component_id": source,
                "all_tests_pass": all_pass, "visible_all_pass": True,
                "hidden_all_pass": all_pass, "test_pass_fraction": 1.0 if all_pass else 0.4,
                "tests": tests, "test_only_fake_execution": True}

    monkeypatch.setattr(ev, "probe_sandbox", fake_probe)
    monkeypatch.setattr(ev, "run_stage", fake_generation_and_real_evaluation)
    monkeypatch.setattr(ev, "_evaluate_candidate", fake_candidate_backend)
    finished = ev.run(case)
    assert stages == ["generate", "evaluate"]
    assert len(probes) >= 3
    assert len(scored_sources) == len(set(scored_sources)) == 500
    assert finished["status"] == "finished_with_real_heldout"
    report_path = case / "evaluation/heldout/report.json"
    report = ev.read_json(report_path)
    complete = ev.read_json(case / "evaluation/complete.json")
    assert complete["heldout_report_sha256"] == finished["heldout_report_sha256"] == ev.file_sha(report_path)
    assert complete["frozen_plan_sha256"] == report["frozen_plan_sha256"] == plan["frozen_plan_sha256"]
    assert report["generation_runtime"]["scientific_evidence"] is False
    assert report["execution_lock_sha256"] == execution_lock
    assert report["generator_dependency_versions"] == versions
    assert report["all_tests_passes"] == 250
    assert report["mean_test_pass_fraction"] == pytest.approx(0.7)

    collector = _load_module("_pipeline_collector", ROOT / "scripts/collect_eesd_rebuttal_case_evals.py")
    destination = tmp_path / "synthetic-collection"
    manifest = collector.collect(cases_root, STUDY, SEED, args["eval_ids"], destination, case_ids=[CASE_ID])
    assert manifest["common_training_parameters"] is None
    assert manifest["arms"]["frozen"]["raw_evaluation_report_sha256"] == ev.file_sha(report_path)
    assert manifest["evaluation_identity"]["generator_dependency_versions"] == versions
    assert (destination / "problem-map.jsonl").read_bytes() == args["eval_ids"].read_bytes()

    summary_module = _load_module("_pipeline_summary", ROOT / "scripts/summarize_eesd_rebuttal_cases.py")
    summary = summary_module.summarize(destination, draws=200, seed=314159)
    assert summary["case_ids"] == [CASE_ID]
    assert summary["source_count"] == 500 and summary["distinct_problem_count"] == 100
    baseline = summary["cases"][CASE_ID]
    assert baseline["training"] is None
    assert baseline["source_all_tests_pass_rate"] == pytest.approx(0.5)
    assert baseline["source_mean_test_pass_rate"] == pytest.approx(0.7)
    assert baseline["problem_mean_all_tests_pass"]["estimate"] == pytest.approx(0.5)
    assert baseline["problem_mean_test_pass_rate"]["estimate"] == pytest.approx(0.7)
    interval = baseline["problem_mean_all_tests_pass"]["ci95"]
    assert interval[0] < 0.5 < interval[1]
    assert summary["contrasts"]["adaptive_vs_frozen"]["status"] == "not_estimable"
    assert summary["contrasts"]["adaptive_vs_frozen"]["missing_case_ids"] == ["C03_EED"]
