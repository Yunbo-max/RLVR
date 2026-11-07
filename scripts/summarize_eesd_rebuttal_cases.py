#!/usr/bin/env python3
"""Summarize a verified independent-case collection with paired problem bootstrap.

This consumes preserved evaluator outcomes only. C00-C03 and explicit case subsets
work independently; absent controls have non-estimable contrasts. Submitted lexical
EED comparisons and the later token extension are labeled separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_eesd_rebuttal_case_evals import (CASE_ALIASES, file_bound, read_json, require,
                                             same, sha, source_problem_map, verify_records)

SCHEMAS = {"eesd-iclr2027-six-arm-heldout-collection-v1", "eesd-iclr2027-case-heldout-collection-v1"}
STATUSES = {"six_real_heldout_reports_verified_not_statistically_interpreted",
            "selected_real_heldout_reports_verified_not_statistically_interpreted"}
IDENTITY = ("model", "revision", "domain", "task_manifest_sha256", "execution_profile",
            "execution_lock_sha256", "sources")


def _contrasts():
    specs = {
        "adaptive_vs_fixed_uniform": ("submitted_lexical_eed", {"C03_EED": 1, "C02_FIXED": -1}),
        "adaptive_vs_equal_uniform": ("submitted_lexical_eed", {"C03_EED": 1, "C01_EQUAL": -1}),
        "equal_vs_frozen": ("submitted_lexical_eed", {"C01_EQUAL": 1, "C00_FROZEN": -1}),
        "adaptive_vs_frozen": ("submitted_lexical_eed", {"C03_EED": 1, "C00_FROZEN": -1}),
    }
    controls = {"constant_mass": "C04_CONSTANT", "shuffled_mass": "C05_SHUFFLED",
                "no_uncertainty": "C06_NO_UNCERTAINTY", "final_filter": "C07_FINAL_FILTER",
                "confidence": "C08_CONFIDENCE", "terminal_eed": "C09_TERMINAL_EED",
                "kappa8": "C13_KAPPA8", "kappa32": "C14_KAPPA32",
                "uniform_relevance": "C15_UNIFORM_RELEVANCE", "permuted_relevance": "C16_PERMUTED_RELEVANCE"}
    for name, case_id in controls.items():
        specs["adaptive_vs_" + name] = ("submitted_lexical_eed_control", {"C03_EED": 1, case_id: -1})
    specs.update({
        "token_edit_with_adaptive_mass": ("token_extension", {"C11_EED_EDIT": 1, "C03_EED": -1}),
        "token_edit_with_fixed_mass": ("token_extension", {"C10_FIXED_EDIT": 1, "C02_FIXED": -1}),
        "adaptive_vs_fixed_edit": ("token_extension", {"C11_EED_EDIT": 1, "C10_FIXED_EDIT": -1}),
        "token_edit_vs_random": ("token_extension", {"C11_EED_EDIT": 1, "C12_EED_RANDOM_TOKEN": -1}),
        "token_mass_interaction": ("token_extension", {"C11_EED_EDIT": 1, "C10_FIXED_EDIT": -1,
                                                        "C03_EED": -1, "C02_FIXED": 1}),
    })
    return specs


def _mapping_from_report(report):
    mapping = {}
    for row in report.get("records", []):
        sid, pid = row.get("source_component_id"), row.get("problem_id")
        if not isinstance(sid, str) or not sid or isinstance(pid, (bool, list, dict)) or pid is None or not str(pid):
            raise ValueError("collected records require a real source/problem mapping")
        if sid in mapping:
            raise ValueError("duplicate source/problem mapping")
        mapping[sid] = str(pid)
    if not mapping:
        raise ValueError("empty collected source/problem mapping")
    return mapping


def _load(collection):
    manifest_path = collection / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema") not in SCHEMAS or manifest.get("status") not in STATUSES:
        raise ValueError("a verified independent-case collection is required")
    arms = require(manifest, "arms", "collection")
    if not isinstance(arms, dict) or not arms:
        raise ValueError("collection has no cases")
    identity = require(manifest, "evaluation_identity", "collection")
    reports, details, first_mapping, first_tests = {}, {}, None, None
    if "problem_map_file" in manifest:
        name = manifest["problem_map_file"]
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("invalid collected problem map filename")
        file_bound(collection / name, manifest.get("problem_map_sha256"), "frozen problem mapping")
        first_mapping = source_problem_map(collection / name)
    for alias, entry in arms.items():
        case_id = require(entry, "case_id", "collection case")
        if case_id in reports or case_id not in CASE_ALIASES or CASE_ALIASES[case_id] != alias:
            raise ValueError("invalid or duplicate collected case identity")
        filename = require(entry, "report_file", "collection report")
        if filename != alias + ".json":
            raise ValueError("invalid collected report filename")
        path = collection / filename
        file_bound(path, entry.get("collected_sha256"), "collected report checksum")
        report = read_json(path)
        same(report.get("schema"), "eesd-fresh-policy-eval-v1", "report identity schema")
        for key in IDENTITY:
            same(report.get(key), identity.get(key), f"report identity {key}")
        for key in ("evaluation_protocol", "generation_policy", "dependency_versions", "generator_dependency_versions"):
            if key in report:
                same(report[key], identity.get(key), f"report identity {key}")
        same(report.get("adapter_sha256"), entry.get("adapter_tree_sha256"), "report identity adapter")
        if "seed" in report:
            same(report["seed"], manifest["seed"], "report identity seed")
        mapping = _mapping_from_report(report)
        if first_mapping is None:
            first_mapping = mapping
        same(mapping, first_mapping, "source/problem mapping differs")
        tests = verify_records(report, first_mapping)
        if first_tests is None:
            first_tests = tests
        same(tests, first_tests, "paired evaluator test inventory differs")
        reports[case_id], details[case_id] = report, entry
    same(set(require(manifest, "case_ids", "collection")), set(reports), "collection case inventory")
    same(len(manifest["case_ids"]), len(reports), "collection duplicate case inventory")
    same(manifest.get("source_count"), len(first_mapping), "collection source count")
    same(manifest.get("distinct_problem_count"), len(set(first_mapping.values())), "collection problem count")
    same(identity.get("problem_map_sha256"), manifest.get("problem_map_sha256"), "collection frozen problem mapping identity")
    return manifest, reports, details, first_mapping


def summarize(collection: Path, draws: int = 10000, seed: int = 314159) -> dict:
    """Return conditional paired inference for exactly the cases in a sealed collection."""
    if type(draws) is not int or draws < 100:
        raise ValueError("bootstrap draws must be an integer at least 100")
    if type(seed) is not int or seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    collection = Path(collection)
    manifest, reports, details, mapping = _load(collection)
    case_ids = sorted(reports)
    source_ids, problems = sorted(mapping), sorted(set(mapping.values()))
    source_index = {sid: index for index, sid in enumerate(source_ids)}
    # Source outcomes are kept separate from equal-weight problem means.
    source_values = np.empty((len(source_ids), len(case_ids), 2), dtype=float)
    for column, case_id in enumerate(case_ids):
        for row in reports[case_id]["records"]:
            source_values[source_index[row["source_component_id"]], column] = [
                float(row["all_tests_pass"]),
                sum(test["outcome"] == "PASS" for test in row["tests"]) / len(row["tests"]),
            ]
    problem_values = np.stack([source_values[[source_index[sid] for sid in source_ids if mapping[sid] == pid]].mean(axis=0)
                               for pid in problems])
    observed = problem_values.mean(axis=0)
    rng = np.random.default_rng(seed)
    boot = np.empty((draws, len(case_ids), 2), dtype=float)
    # One shared set of draws for every case and contrast preserves pairing; batches
    # bound the large sampled intermediate for full 17-case / 500-problem studies.
    for start in range(0, draws, 128):
        end = min(draws, start + 128)
        indices = rng.integers(0, len(problems), size=(end - start, len(problems)))
        boot[start:end] = problem_values[indices].mean(axis=1)

    def metric(estimate, samples):
        return {"estimate": float(estimate), "ci95": [float(x) for x in np.quantile(samples, [.025, .975])]}

    cases = {}
    for column, case_id in enumerate(case_ids):
        entry = details[case_id]
        cases[case_id] = {
            "alias": CASE_ALIASES[case_id], "case_specification": entry["case_specification"],
            "source_count": len(source_ids), "distinct_problem_count": len(problems),
            "all_tests_passes": int(source_values[:, column, 0].sum()),
            "source_all_tests_pass_rate": float(source_values[:, column, 0].mean()),
            "source_mean_test_pass_rate": float(source_values[:, column, 1].mean()),
            "problem_mean_all_tests_pass": metric(observed[column, 0], boot[:, column, 0]),
            "problem_mean_test_pass_rate": metric(observed[column, 1], boot[:, column, 1]),
            "training": entry["training"], "adapter_tree_sha256": entry["adapter_tree_sha256"],
        }
    contrasts = {}
    for name, (group, coefficients) in _contrasts().items():
        missing = sorted(set(coefficients) - set(case_ids))
        value = {"group": group, "coefficients": coefficients,
                 "status": "not_estimable" if missing else "estimated"}
        if missing:
            value["missing_case_ids"] = missing
        else:
            weights = np.array([coefficients.get(case_id, 0) for case_id in case_ids])
            estimate, samples = weights @ observed, np.einsum("c,dcm->dm", weights, boot)
            source_delta = np.einsum("c,scm->sm", weights, source_values).mean(axis=0)
            value.update(problem_mean_all_tests_pass=metric(estimate[0], samples[:, 0]),
                         problem_mean_test_pass_rate=metric(estimate[1], samples[:, 1]),
                         source_all_tests_pass_difference=float(source_delta[0]),
                         source_mean_test_pass_rate_difference=float(source_delta[1]))
            if len(coefficients) == 2 and sorted(coefficients.values()) == [-1, 1]:
                intervention = case_ids.index(next(k for k, w in coefficients.items() if w == 1))
                reference = case_ids.index(next(k for k, w in coefficients.items() if w == -1))
                delta = source_values[:, intervention, 0] - source_values[:, reference, 0]
                fixes, regressions = int((delta > 0).sum()), int((delta < 0).sum())
                value.update(source_fixes=fixes, source_regressions=regressions,
                             source_net_fixes=fixes - regressions,
                             sources_unchanged=int((delta == 0).sum()))
        contrasts[name] = value
    return {
        "schema": "eesd-iclr2027-case-heldout-paired-summary-v1",
        "status": "existing_heldout_outcomes_summarized_no_new_generation",
        "study": manifest["study"], "training_seed": manifest["seed"],
        "source_count": len(source_ids), "distinct_problem_count": len(problems),
        "case_ids": case_ids, "evaluation_identity": manifest["evaluation_identity"],
        "common_training_parameters": manifest["common_training_parameters"],
        "estimand": "Mean across distinct heldout problems of each problem's mean across its source components. Test-pass rates first average the tests within each source; source-weighted descriptive rates are also reported.",
        "uncertainty": {
            "method": "paired problem-cluster percentile bootstrap", "draws": draws,
            "seed": seed, "confidence_level": .95,
            "scope": "Intervals are conditional on these fitted models, their training seed, and the preserved heldout population. They do not estimate variability from retraining or establish exact historical replication.",
            "multiplicity": "Intervals are descriptive and unadjusted across the named contrasts.",
            "degenerate_single_problem": len(problems) == 1,
        },
        "provenance": {"collection_manifest_sha256": sha(collection / "manifest.json"),
                       "problem_map_sha256": manifest["problem_map_sha256"],
                       "corrections_sha256": manifest["corrections_sha256"],
                       "case_receipts": {case_id: details[case_id] for case_id in case_ids}},
        "cases": cases, "contrasts": contrasts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", "--eval-dir", dest="collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", "--seed", dest="bootstrap_seed", type=int, default=314159)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("summary output already exists")
    result = summarize(args.collection, args.draws, args.bootstrap_seed)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(text)
    print(json.dumps({key: result[key] for key in ("status", "case_ids", "source_count", "distinct_problem_count")}, indent=2))


if __name__ == "__main__":
    main()
