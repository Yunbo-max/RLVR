"""Synthetic local receipts exercise verification, never scientific outcomes."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from eesd.training_outcomes import tree_sha

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "independent_collector", ROOT / "scripts/collect_eesd_rebuttal_case_evals.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
P0 = ["C00_FROZEN", "C01_EQUAL", "C02_FIXED", "C03_EED"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True) + "\n")


def _fixture(tmp_path, ids=None):
    ids = list(m.ARMS.values()) if ids is None else ids
    registry = json.loads((ROOT / "configs/rebuttal/eesd_case_registry.json").read_text())
    specs = {c["id"]: c for c in registry["cases"]}
    study, seed = "unit_test", 1701
    model = "Qwen/Qwen2.5-Coder-7B-Instruct"
    model_file = tmp_path / "model.yaml"
    model_file.write_text(f"model_id: {model}\nrevision: {'a'*40}\n")
    parent = tmp_path / "initial"
    write(parent / "adapter_config.json", {"r": 8, "lora_alpha": 16})
    (parent / "adapter_model.safetensors").write_bytes(b"synthetic initial weights")
    path = tmp_path / "map.jsonl"
    path.write_text("".join(json.dumps({"source_component_id": s, "problem_id": p})+"\n"
                            for s, p in [("s1", "p1"), ("s2", "p2"), ("s3", "p2")]))
    corrections = tmp_path / "corrections.jsonl"
    corrections.write_text(json.dumps({"source_component_id": "train1", "problem_id": "trainp1",
                                       "split": "train", "trajectory_id": "t1"})+"\n")
    public, private = tmp_path / "public", tmp_path / "private"
    for folder in (public, private):
        folder.mkdir()
        (folder / "tasks.jsonl").write_text("".join(json.dumps({
            "source_component_id": s, "split": "primary"})+"\n" for s in ("s1", "s2", "s3")))
    write(public / "manifest.json", {"schema": "apbpf-rbr-generated-materialization-v1", "public_tasks_sha256": digest(public / "tasks.jsonl")})
    write(private / "manifest.json", {"schema": "apbpf-rbr-generated-materialization-v1", "public_tasks_sha256": digest(public / "tasks.jsonl"),
                                      "evaluator_tasks_sha256": digest(private / "tasks.jsonl")})
    for i, case_id in enumerate(ids):
        folder = tmp_path / study / case_id / f"seed{seed}"
        case_spec = specs[case_id]
        baseline = case_spec["kind"] == "baseline"
        setup = {
            "model_config_path": str(model_file), "model_config_sha256": digest(model_file),
            "shared_init_adapter_path": str(parent), "shared_init_adapter_tree_sha256": tree_sha(parent),
        }
        prov = {"schema": "eesd-token-aware-input-readiness-v1", "errors": [],
                "provenance_status": "problem_ids_disjoint_checked",
                "correction_file_sha256": digest(corrections),
                "missing_problem_id_counts": {"train": 0, "development": 0},
                "evaluation": {"eval_input_sha256": digest(path), "eval_rows": 3,
                               "eval_source_ids_supplied": 3, "eval_problem_ids_supplied": 2,
                               "source_level_overlap_checked": True, "problem_level_overlap_checked": True}}
        write(folder / "data_provenance.json", prov)
        frozen = {"eval_ids_sha256": digest(path)}
        case = {"schema": "eesd-iclr2027-independent-case-v1", "study": study, "seed": seed,
                "case": case_spec, "setup": setup, "frozen_case_input_hashes": frozen,
                "data_provenance_sha256": digest(folder / "data_provenance.json"),
                "status": "completed_diagnostic_without_gpu" if baseline else "trained_awaiting_independent_evaluation"}
        adapter = parent
        if not baseline:
            setup.update({"model_id": model, "revision": "a"*40, "response_token_budget": 512,
                          "seed": seed, "corrections_path": str(corrections),
                          "corrections_sha256": digest(corrections), "trainer_source_sha256": "e"*64,
                          "max_steps": 200, "gradient_accumulation": 16, "max_length": 4608,
                          "learning_rate": 0.0002, "beta": 0.03, "alpha": 0.1,
                          "lambda_penalty": 0.5, "token_edit_boost": 2.0,
                          "optimizer": {"name": "torch.optim.AdamW", "learning_rate": 0.0002},
                          "lora_config": {"r": 8, "lora_alpha": 16}})
            scored = folder / "scored-corrections.jsonl"
            scored.write_bytes(corrections.read_bytes())
            frozen["scored-corrections.jsonl"] = digest(scored)
            adapter = folder / "model" / "adapter"
            write(adapter / "adapter_config.json", {"r": 8, "lora_alpha": 16})
            (adapter / "adapter_model.safetensors").write_bytes(f"synthetic weights {case_id}".encode())
            training = {"schema": "eesd-weighted-sft-v1", "rule": "eesd_full",
                        "input_sha256": digest(scored), "response_tokens": 512, "anchor_beta": 0.03,
                        "include_zero_weight_kl": True, "token_weighting": case_spec["token"],
                        "previous_adapter_tree_sha256_before": tree_sha(parent),
                        "previous_adapter_tree_sha256_after": tree_sha(parent),
                        "training_schedule_schema": "eesd-scheduled-response-tokens-v1",
                        "training_schedule_sha256": "9"*64,
                        "loss_normalization": "response_tokens", "micro_steps": 16,
                        "optimizer_steps": 1, "scheduled_training_trajectories": 1}
            for key in ("model_id", "revision", "model_config_sha256", "seed", "response_token_budget",
                        "trainer_source_sha256", "token_edit_boost", "max_steps", "gradient_accumulation",
                        "max_length", "learning_rate", "optimizer", "lora_config"):
                training[key] = setup[key]
            write(folder / "model" / "training-report.json", training)
            case["training_report_sha256"] = digest(folder / "model" / "training-report.json")
            case["trained_adapter_tree_sha256"] = tree_sha(adapter)
        write(folder / "case.json", case)
        rows = []
        for j, sid in enumerate(("s1", "s2", "s3")):
            success = j == 0 or bool(i % 2)
            rows.append({"source_component_id": sid, "task_id": "task-"+sid,
                         "all_tests_pass": success,
                         "tests": [{"test_id": "0", "outcome": "PASS"},
                                   {"test_id": "4", "outcome": "PASS" if success else "FAIL"}]})
        passes = sum(row["all_tests_pass"] for row in rows)
        report = {"schema": "eesd-fresh-policy-eval-v1", "model": model, "revision": "a"*40,
                  "domain": "rbr", "adapter_sha256": tree_sha(adapter),
                  "task_manifest_sha256": digest(private / "manifest.json"),
                  "execution_profile": "sandbox", "execution_lock_sha256": None,
                  "sources": 3, "all_tests_passes": passes,
                  "fresh_all_tests_pass_at_1": passes / 3, "records": rows}
        result = folder / "evaluation" / "heldout" / "report.json"
        write(result, report)
        plan = {"schema": "eesd-rebuttal-per-case-original-evaluator-v1",
                "status": "finished_with_real_heldout", "case_sha256": digest(folder / "case.json"),
                "case_id": case_id, "study": study, "seed": seed,
                "model_id": model, "revision": "a"*40, "domain": "rbr",
                "model_config_path": str(model_file), "model_config_sha256": digest(model_file),
                "adapter_path": str(adapter), "adapter_tree_sha256": tree_sha(adapter),
                "data_provenance_sha256": digest(folder / "data_provenance.json"),
                "eval_source_id_file_path": str(path), "eval_source_id_file_sha256": digest(path),
                "source_disjointness_status": "problem_ids_disjoint_checked",
                "evaluation_protocol": "original-pbpf-greedy-sandbox-v1", "pbpf_commit": "0"*40,
                "generator_sha256": "1"*64, "evaluator_sha256": "2"*64,
                "public_root": str(public), "evaluator_root": str(private),
                "public_manifest_sha256": digest(public / "manifest.json"),
                "public_tasks_sha256": digest(public / "tasks.jsonl"),
                "evaluator_manifest_sha256": digest(private / "manifest.json"),
                "evaluator_tasks_sha256": digest(private / "tasks.jsonl"),
                "evaluation_source_ids": ["s1", "s2", "s3"], "evaluation_sources": 3,
                "heldout_report_sha256": digest(result), "heldout_count": 3,
                "heldout_all_tests_pass_at_1": passes / 3}
        write(folder / "evaluation" / "plan.json", plan)
    return study, path


def _mutate(tmp_path, study, case_id, target, update):
    folder = tmp_path / study / case_id / "seed1701"
    path = folder / target
    value = json.loads(path.read_text())
    update(value)
    write(path, value)
    if target == "model/training-report.json":
        case_path = folder / "case.json"
        case = json.loads(case_path.read_text())
        case["training_report_sha256"] = digest(path)
        write(case_path, case)
    plan_path = folder / "evaluation" / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["case_sha256"] = digest(folder / "case.json")
    plan["heldout_report_sha256"] = digest(folder / "evaluation" / "heldout" / "report.json")
    write(plan_path, plan)


def test_collect_six_cases_retains_positional_api_aliases(tmp_path):
    study, path = _fixture(tmp_path)
    output = tmp_path / "out"
    manifest = m.collect(tmp_path, study, 1701, path, output)
    assert manifest["schema"] == "eesd-iclr2027-six-arm-heldout-collection-v1"
    assert set(manifest["arms"]) == set(m.ARMS)
    assert manifest["distinct_problem_count"] == 2
    report = json.loads((output / "adaptive_edit.json").read_text())
    assert report["records"][2]["problem_id"] == "p2"
    with pytest.raises(FileExistsError):
        m.collect(tmp_path, study, 1701, path, output)


def test_p0_accepts_frozen_baseline_sparse_setup_without_other_cases(tmp_path):
    study, path = _fixture(tmp_path, P0)
    manifest = m.collect(tmp_path, study, 1701, path, tmp_path / "out", case_set="p0")
    assert {x["case_id"] for x in manifest["arms"].values()} == set(P0)
    assert manifest["arms"]["frozen"]["training"] is None
    assert manifest["common_training_parameters"]["training_schedule_sha256"] == "9"*64


@pytest.mark.parametrize("errors", [{}, {"missing_problem_id": 1}])
def test_actual_input_audit_error_counter_is_supported(tmp_path, errors):
    study, path = _fixture(tmp_path, ["C00_FROZEN"])
    folder = tmp_path / study / "C00_FROZEN" / "seed1701"
    provenance_path = folder / "data_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["errors"] = errors
    write(provenance_path, provenance)
    case_path = folder / "case.json"
    case = json.loads(case_path.read_text())
    case["data_provenance_sha256"] = digest(provenance_path)
    write(case_path, case)
    plan_path = folder / "evaluation/plan.json"
    plan = json.loads(plan_path.read_text())
    plan.update(data_provenance_sha256=digest(provenance_path), case_sha256=digest(case_path))
    write(plan_path, plan)
    if errors:
        with pytest.raises(ValueError, match="audit errors"):
            m.collect(tmp_path, study, 1701, path, tmp_path / "out", case_ids=["C00_FROZEN"])
    else:
        result = m.collect(tmp_path, study, 1701, path, tmp_path / "out", case_ids=["C00_FROZEN"])
        assert result["case_ids"] == ["C00_FROZEN"]


def test_all_cases_collects_every_registered_independent_case(tmp_path):
    study, path = _fixture(tmp_path, list(m.CASE_ALIASES))
    manifest = m.collect(tmp_path, study, 1701, path, tmp_path / "out", case_set="all")
    assert manifest["case_ids"] == list(m.CASE_ALIASES)
    assert {item["case_id"] for item in manifest["arms"].values()} == set(m.CASE_ALIASES)
    assert len(manifest["arms"]) == 17
    assert manifest["arms"]["frozen"]["training"] is None
    assert manifest["arms"]["adaptive_edit"]["training"]["token_weighting"] == "edit"


def test_explicit_subset_only_requires_requested_case(tmp_path):
    study, path = _fixture(tmp_path, ["C03_EED"])
    result = m.collect(tmp_path, study, 1701, path, tmp_path / "out", case_ids=["C03_EED"])
    assert list(result["arms"]) == ["adaptive_uniform"]


@pytest.mark.parametrize("field,value", [("domain", "codearc"), ("model", "wrong"),
                                        ("adapter_sha256", "f"*64), ("task_manifest_sha256", "f"*64)])
def test_report_cannot_replace_plan_identity_even_if_all_selected_reports_match(tmp_path, field, value):
    study, path = _fixture(tmp_path, ["C03_EED"])
    _mutate(tmp_path, study, "C03_EED", "evaluation/heldout/report.json", lambda d: d.update({field: value}))
    with pytest.raises(ValueError, match="evaluation protocol|report binding"):
        m.collect(tmp_path, study, 1701, path, tmp_path / "out", case_ids=["C03_EED"])
    assert not (tmp_path / "out").exists()


def test_changed_problem_cluster_map_is_rejected_even_with_same_sources(tmp_path):
    study, path = _fixture(tmp_path)
    changed = tmp_path / "posthoc.jsonl"
    changed.write_text(path.read_text().replace('"p2"', '"p1"'))
    with pytest.raises(ValueError, match="frozen.*map"):
        m.collect(tmp_path, study, 1701, changed, tmp_path / "out")


@pytest.mark.parametrize("field,value", [("response_tokens", 511), ("seed", 42),
                                        ("gradient_accumulation", 8), ("token_weighting", "random"),
                                        ("previous_adapter_tree_sha256_after", "c"*64)])
def test_actual_training_report_must_match_prepared_case(tmp_path, field, value):
    study, path = _fixture(tmp_path)
    _mutate(tmp_path, study, "C03_EED", "model/training-report.json", lambda d: d.update({field: value}))
    with pytest.raises(ValueError, match="training binding"):
        m.collect(tmp_path, study, 1701, path, tmp_path / "out")


def test_different_actual_token_schedule_rejected_across_arms(tmp_path):
    study, path = _fixture(tmp_path)
    _mutate(tmp_path, study, "C03_EED", "model/training-report.json",
            lambda d: d.update(training_schedule_sha256="8"*64))
    with pytest.raises(ValueError, match="training.*schedule"):
        m.collect(tmp_path, study, 1701, path, tmp_path / "out")


def test_actual_training_dependency_versions_retained_with_frozen_baseline(tmp_path):
    study, path = _fixture(tmp_path, P0)
    versions = {"torch": "2.5.1", "transformers": "4.48.0", "peft": "0.14.0", "bitsandbytes": "0.45.0"}
    for case_id in P0[1:]:
        _mutate(tmp_path, study, case_id, "model/training-report.json",
                lambda d: d.update(dependency_versions=versions))
    manifest = m.collect(tmp_path, study, 1701, path, tmp_path / "out", case_set="p0")
    assert manifest["common_training_parameters"]["dependency_versions"] == versions
    assert manifest["arms"]["frozen"]["training"] is None
    for alias in ("equal_uniform", "fixed_uniform", "adaptive_uniform"):
        assert manifest["arms"][alias]["training"]["dependency_versions"] == versions


@pytest.mark.parametrize("change", ("different", "missing_first", "missing_last"))
def test_actual_training_dependency_versions_must_agree_across_cases(tmp_path, change):
    ids = ["C01_EQUAL", "C03_EED"]
    study, path = _fixture(tmp_path, ids)
    versions = {"torch": "2.5.1", "transformers": "4.48.0", "peft": "0.14.0", "bitsandbytes": "0.45.0"}
    for index, case_id in enumerate(ids):
        if (change == "missing_first" and index == 0) or (change == "missing_last" and index == 1):
            continue
        actual = {**versions, "torch": "2.6.0"} if change == "different" and index == 1 else versions
        _mutate(tmp_path, study, case_id, "model/training-report.json",
                lambda d, actual=actual: d.update(dependency_versions=actual))
    with pytest.raises(ValueError, match="training.*dependency_versions"):
        m.collect(tmp_path, study, 1701, path, tmp_path / "out", case_ids=ids)
    assert not (tmp_path / "out").exists()


def test_all_tests_summary_must_follow_raw_test_outcomes(tmp_path):
    study, path = _fixture(tmp_path)
    def change(report):
        report["records"][0]["tests"][0]["outcome"] = "FAIL"
    _mutate(tmp_path, study, "C03_EED", "evaluation/heldout/report.json", change)
    with pytest.raises(ValueError, match="test outcomes"):
        m.collect(tmp_path, study, 1701, path, tmp_path / "out")


def test_case_receipt_changed_after_evaluation_prepared_rejected(tmp_path):
    study, path = _fixture(tmp_path)
    case_path = tmp_path / study / "C03_EED" / "seed1701" / "case.json"
    case = json.loads(case_path.read_text())
    case["study"] = "other-study"
    write(case_path, case)
    with pytest.raises(ValueError, match="receipt|case binding"):
        m.collect(tmp_path, study, 1701, path, tmp_path / "out")


def _as_matched(tmp_path, study, case_id):
    folder = tmp_path / study / case_id / 'seed1701'
    plan_path = folder / 'evaluation/plan.json'
    plan = json.loads(plan_path.read_text())
    plan.update(schema='eesd-rebuttal-per-case-matched-evaluator-v1',
                evaluation_protocol='eesd-rebuttal-matched-prospective-v1',
                local_source_hashes={'src/eesd/rebuttal_eval.py': 'a' * 64},
                pbpf_source_hashes={'src/pbpf/apbpf/rbr_execution.py': 'b' * 64},
                generation_policy={'do_sample': False, 'num_beams': 1, 'quantization': 'nf4-bf16'})
    report_path = folder / 'evaluation/heldout/report.json'
    report = json.loads(report_path.read_text())
    report.update(evaluation_protocol=plan['evaluation_protocol'], generation_policy=plan['generation_policy'])
    write(report_path, report)
    plan['heldout_report_sha256'] = digest(report_path)
    write(plan_path, plan)
    return plan_path, report_path


def test_suite_cli_alias_runs_independent_p0(tmp_path, monkeypatch, capsys):
    import sys
    study, path = _fixture(tmp_path, P0)
    monkeypatch.setattr(sys, 'argv', ['collector', '--cases-root', str(tmp_path), '--study', study,
        '--seed', '1701', '--problem-map', str(path), '--output', str(tmp_path / 'out'), '--suite', 'p0'])
    m.main()
    assert set(json.loads(capsys.readouterr().out)['case_ids']) == set(P0)


@pytest.mark.parametrize('field,value', [
    ('schema', 'invented-matched-plan-v1'),
    ('evaluation_protocol', 'invented-prospective-protocol-v1'),
])
def test_only_exact_registered_prospective_plan_is_accepted(tmp_path, field, value):
    study, path = _fixture(tmp_path, ['C03_EED'])
    plan_path, _ = _as_matched(tmp_path, study, 'C03_EED')
    plan = json.loads(plan_path.read_text())
    plan[field] = value
    write(plan_path, plan)
    with pytest.raises(ValueError, match='evaluation protocol'):
        m.collect(tmp_path, study, 1701, path, tmp_path / 'out', case_ids=['C03_EED'])


def test_actual_generation_policy_must_match_prospective_plan(tmp_path):
    study, path = _fixture(tmp_path, ['C03_EED'])
    _as_matched(tmp_path, study, 'C03_EED')
    _mutate(tmp_path, study, 'C03_EED', 'evaluation/heldout/report.json',
            lambda d: d.update(generation_policy={'do_sample': True}))
    with pytest.raises(ValueError, match='generation.policy|generation policy'):
        m.collect(tmp_path, study, 1701, path, tmp_path / 'out', case_ids=['C03_EED'])


def test_matched_collection_retains_frozen_evaluator_source_maps(tmp_path):
    study, path = _fixture(tmp_path, ['C01_EQUAL', 'C03_EED'])
    for case_id in ['C01_EQUAL', 'C03_EED']:
        _as_matched(tmp_path, study, case_id)
    result = m.collect(tmp_path, study, 1701, path, tmp_path / 'out', case_ids=['C01_EQUAL', 'C03_EED'])
    assert result['evaluation_identity']['local_source_hashes'] == {'src/eesd/rebuttal_eval.py': 'a' * 64}
    assert result['evaluation_identity']['pbpf_source_hashes'] == {'src/pbpf/apbpf/rbr_execution.py': 'b' * 64}


@pytest.mark.parametrize('field', ['local_source_hashes', 'pbpf_source_hashes'])
@pytest.mark.parametrize('change', ['missing', 'different'])
def test_matched_evaluator_source_maps_cannot_be_missing_or_differ(tmp_path, field, change):
    ids = ['C01_EQUAL', 'C03_EED']
    study, path = _fixture(tmp_path, ids)
    for case_id in ids:
        plan_path, _ = _as_matched(tmp_path, study, case_id)
    plan = json.loads(plan_path.read_text())
    if change == 'missing':
        plan.pop(field)
    else:
        plan[field] = {name: 'c' * 64 for name in plan[field]}
    write(plan_path, plan)
    with pytest.raises(ValueError, match='source|evaluation protocol'):
        m.collect(tmp_path, study, 1701, path, tmp_path / 'out', case_ids=ids)
    assert not (tmp_path / 'out').exists()


def test_actual_generator_dependency_versions_cannot_differ_across_cases(tmp_path):
    study, path = _fixture(tmp_path, P0)
    for case_id in P0:
        _mutate(tmp_path, study, case_id, 'evaluation/heldout/report.json',
                lambda d, c=case_id: d.update(generator_dependency_versions={
                    'transformers': '4.50.0' if c == 'C03_EED' else '4.49.0'}))
    with pytest.raises(ValueError, match='evaluation protocol|dependency'):
        m.collect(tmp_path, study, 1701, path, tmp_path / 'out', case_set='p0')


def test_jsonl_records_preserve_literal_unicode_line_separators(tmp_path):
    path = tmp_path / 'map.jsonl'
    path.write_bytes((json.dumps({'source_component_id': 's\u0085one', 'problem_id': 'p\u2028one'},
                                 ensure_ascii=False) + '\r\n').encode())
    assert m.source_problem_map(path) == {'s\u0085one': 'p\u2028one'}
    (tmp_path / 'banks').mkdir()
    study, _ = _fixture(tmp_path / 'banks', ['C03_EED'])
    plan_path = tmp_path / 'banks' / study / 'C03_EED/seed1701/evaluation/plan.json'
    plan = json.loads(plan_path.read_text())
    for root_key, prefix in [('public_root', 'public'), ('evaluator_root', 'evaluator')]:
        folder = Path(plan[root_key])
        tasks = folder / 'tasks.jsonl'
        rows = [json.loads(line) for line in tasks.read_text().split('\n') if line]
        for row in rows:
            row['prompt'] = 'literal NEL \u0085 and LS \u2028 remain inside a JSON string'
        tasks.write_bytes(''.join(json.dumps(row, ensure_ascii=False) + '\r\n' for row in rows).encode())
        plan[f'{prefix}_tasks_sha256'] = digest(tasks)
    for root_key, prefix in [('public_root', 'public'), ('evaluator_root', 'evaluator')]:
        manifest_path = Path(plan[root_key]) / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest.update(public_tasks_sha256=plan['public_tasks_sha256'])
        manifest[f'{prefix}_tasks_sha256'] = plan[f'{prefix}_tasks_sha256']
        write(manifest_path, manifest)
        plan[f'{prefix}_manifest_sha256'] = digest(manifest_path)
    m._bank_identity(plan, {'s1': 'p1', 's2': 'p2', 's3': 'p2'})


def _replay_plan(tmp_path, study):
    plan_path, report_path = _as_matched(tmp_path, study, 'C03_EED')
    plan = json.loads(plan_path.read_text())
    bundle = tmp_path / 'replay'
    for old_key, label, prefix in [('public_root', 'public', 'public'),
                                    ('evaluator_root', 'private', 'evaluator')]:
        folder = bundle / 'apps_replay' / label
        folder.mkdir(parents=True)
        (folder / 'tasks.jsonl').write_bytes((Path(plan[old_key]) / 'tasks.jsonl').read_bytes())
        write(folder / 'manifest.json', {'schema': f'eesd-replay-{prefix}-v1',
              'domain': 'apps_replay', 'task_kind': 'stdin_synthesis',
              'tasks': {'sha256': digest(folder / 'tasks.jsonl'),
                        'bytes': (folder / 'tasks.jsonl').stat().st_size}})
        plan[old_key] = str(folder)
        plan[f'{prefix}_manifest_sha256'] = digest(folder / 'manifest.json')
    files = [Path(plan[key]) / name for key in ['public_root', 'evaluator_root']
             for name in ['manifest.json', 'tasks.jsonl']]
    write(bundle / 'admission.json', {'schema': 'eesd-replay-admission-v1',
          'status': 'data-admitted-not-generated-not-scored',
          'outputs': {p.relative_to(bundle).as_posix(): {'sha256': digest(p), 'bytes': p.stat().st_size}
                      for p in files}})
    plan.update(domain='apps_replay', admission_path=str(bundle / 'admission.json'),
                admission_sha256=digest(bundle / 'admission.json'))
    report = json.loads(report_path.read_text())
    report.update(domain='apps_replay', task_manifest_sha256=plan['evaluator_manifest_sha256'])
    write(report_path, report)
    plan['heldout_report_sha256'] = digest(report_path)
    write(plan_path, plan)
    return plan


def test_replay_nested_manifest_and_admission_supported(tmp_path):
    study, path = _fixture(tmp_path, ['C03_EED'])
    plan = _replay_plan(tmp_path, study)
    result = m.collect(tmp_path, study, 1701, path, tmp_path / 'out', case_ids=['C03_EED'])
    assert result['evaluation_identity']['domain'] == 'apps_replay'
    assert result['evaluation_identity']['admission_sha256'] == plan['admission_sha256']


def test_replay_admission_receipt_must_match_source_files(tmp_path):
    study, path = _fixture(tmp_path, ['C03_EED'])
    plan = _replay_plan(tmp_path, study)
    admission_path = Path(plan['admission_path'])
    admission = json.loads(admission_path.read_text())
    admission['outputs']['apps_replay/private/tasks.jsonl']['sha256'] = 'f' * 64
    write(admission_path, admission)
    plan['admission_sha256'] = digest(admission_path)
    write(tmp_path / study / 'C03_EED/seed1701/evaluation/plan.json', plan)
    with pytest.raises(ValueError, match='Replay admission'):
        m.collect(tmp_path, study, 1701, path, tmp_path / 'out', case_ids=['C03_EED'])


def test_frozen_problem_map_bytes_copied_into_collection(tmp_path):
    study, path = _fixture(tmp_path, P0)
    result = m.collect(tmp_path, study, 1701, path, tmp_path / 'out', case_set='p0')
    assert (tmp_path / 'out' / result['problem_map_file']).read_bytes() == path.read_bytes()
