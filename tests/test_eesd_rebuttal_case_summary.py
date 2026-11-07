"""Synthetic summaries verify statistics and receipts, not benchmark performance."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
fixture_spec = importlib.util.spec_from_file_location(
    "collector_fixtures", ROOT / "tests/test_eesd_rebuttal_case_collector.py")
f = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(f)


def summary_module():
    spec = importlib.util.spec_from_file_location(
        "case_summary", ROOT / "scripts/summarize_eesd_rebuttal_cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collection(tmp_path, ids=None, controlled=False):
    ids = f.P0 if ids is None else ids
    study, mapping = f._fixture(tmp_path, ids)
    if controlled:
        outcomes = {"C00_FROZEN": [[False, False]] * 3,
                    "C01_EQUAL": [[True, False]] * 3,
                    "C02_FIXED": [[False, False], [True, True], [True, True]],
                    "C03_EED": [[True, True], [True, False], [True, False]]}
        for case_id, tests in outcomes.items():
            folder = tmp_path / study / case_id / "seed1701" / "evaluation"
            report_path = folder / "heldout/report.json"
            report = json.loads(report_path.read_text())
            for row, passed in zip(report["records"], tests):
                row["all_tests_pass"] = all(passed)
                for test, passed_test in zip(row["tests"], passed):
                    test["outcome"] = "PASS" if passed_test else "FAIL"
            count = sum(r["all_tests_pass"] for r in report["records"])
            report.update(all_tests_passes=count, fresh_all_tests_pass_at_1=count / 3)
            f.write(report_path, report)
            plan_path = folder / "plan.json"
            plan = json.loads(plan_path.read_text())
            plan.update(heldout_report_sha256=f.digest(report_path),
                        heldout_all_tests_pass_at_1=count / 3)
            f.write(plan_path, plan)
    target = tmp_path / "collection"
    f.m.collect(tmp_path, study, 1701, mapping, target, case_ids=ids)
    return target


def mutate_collected(target, alias, update):
    path = target / (alias + ".json")
    data = json.loads(path.read_text())
    update(data)
    f.write(path, data)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["arms"][alias]["collected_sha256"] = f.digest(path)
    f.write(manifest_path, manifest)


def test_p0_summary_uses_paired_problem_clusters_and_counts_source_fixes(tmp_path):
    target = collection(tmp_path, controlled=True)
    result = summary_module().summarize(target, draws=2000, seed=123)
    assert result["schema"] == "eesd-iclr2027-case-heldout-paired-summary-v1"
    assert result["source_count"] == 3
    assert result["distinct_problem_count"] == 2
    adaptive = result["cases"]["C03_EED"]
    assert adaptive["all_tests_passes"] == 1
    assert adaptive["source_all_tests_pass_rate"] == pytest.approx(1 / 3)
    assert adaptive["problem_mean_all_tests_pass"]["estimate"] == .5
    assert adaptive["source_mean_test_pass_rate"] == pytest.approx(2 / 3)
    assert adaptive["problem_mean_test_pass_rate"]["estimate"] == .75
    contrast = result["contrasts"]["adaptive_vs_fixed_uniform"]
    assert contrast["status"] == "estimated"
    assert contrast["problem_mean_all_tests_pass"]["estimate"] == 0
    assert contrast["problem_mean_all_tests_pass"]["ci95"] == [-1, 1]
    assert contrast["source_all_tests_pass_difference"] == pytest.approx(-1 / 3)
    assert contrast["problem_mean_test_pass_rate"]["estimate"] == .25
    assert contrast["source_fixes"] == 1
    assert contrast["source_regressions"] == 2
    assert contrast["source_net_fixes"] == -1
    missing = result["contrasts"]["token_mass_interaction"]
    assert missing["status"] == "not_estimable"
    assert missing["group"] == "token_extension"
    assert set(missing["missing_case_ids"]) == {"C10_FIXED_EDIT", "C11_EED_EDIT"}
    assert result["provenance"]["collection_manifest_sha256"] == f.digest(target / "manifest.json")
    assert "conditional" in result["uncertainty"]["scope"]


def test_single_case_summary_does_not_demand_uncollected_controls(tmp_path):
    target = collection(tmp_path, ["C03_EED"])
    result = summary_module().summarize(target, draws=100, seed=1)
    assert list(result["cases"]) == ["C03_EED"]
    assert all(c["status"] == "not_estimable" for c in result["contrasts"].values())


def test_summary_rejects_changed_collected_bytes(tmp_path):
    target = collection(tmp_path)
    with (target / "adaptive_uniform.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="checksum"):
        summary_module().summarize(target, draws=100)


def test_summary_rechecks_test_outcomes_after_receipt_update(tmp_path):
    target = collection(tmp_path)
    mutate_collected(target, "adaptive_uniform", lambda d: d["records"][0].update(all_tests_pass=False))
    with pytest.raises(ValueError, match="contradict"):
        summary_module().summarize(target, draws=100)


def test_summary_rejects_changed_problem_assignment_even_if_sources_match(tmp_path):
    target = collection(tmp_path)
    mutate_collected(target, "adaptive_uniform", lambda d: d["records"][0].update(problem_id="other"))
    with pytest.raises(ValueError, match="problem mapping"):
        summary_module().summarize(target, draws=100)


def test_summary_rejects_report_identity_changed_after_collection(tmp_path):
    target = collection(tmp_path)
    mutate_collected(target, "adaptive_uniform", lambda d: d.update(domain="codearc"))
    with pytest.raises(ValueError, match="identity"):
        summary_module().summarize(target, draws=100)


def test_summary_deterministic_and_validates_bootstrap_draws(tmp_path):
    target = collection(tmp_path)
    m = summary_module()
    assert m.summarize(target, draws=100, seed=12) == m.summarize(target, draws=100, seed=12)
    with pytest.raises(ValueError, match="draws"):
        m.summarize(target, draws=0)


def test_six_arm_collection_still_yields_token_interaction(tmp_path):
    target = collection(tmp_path, list(f.m.ARMS.values()))
    result = summary_module().summarize(target, draws=100)
    contrast = result["contrasts"]["token_mass_interaction"]
    assert contrast["status"] == "estimated"
    assert contrast["coefficients"] == {"C11_EED_EDIT": 1, "C10_FIXED_EDIT": -1,
                                         "C03_EED": -1, "C02_FIXED": 1}
    assert "source_fixes" not in contrast


def test_eval_dir_cli_alias(tmp_path, monkeypatch, capsys):
    import sys
    module = summary_module()
    folder = collection(tmp_path)
    output = tmp_path / 'summary.json'
    monkeypatch.setattr(sys, 'argv', ['summary', '--eval-dir', str(folder),
                                    '--output', str(output), '--draws', '100'])
    module.main()
    assert json.loads(output.read_text())['source_count'] == 3
