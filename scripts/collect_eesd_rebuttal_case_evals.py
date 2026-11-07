#!/usr/bin/env python3
"""Verify and collect independently completed heldout cases; never create outcomes.

The default preserves the original six token-arm aliases and positional Python API.
Use --suite p0 for C00-C03, --suite all for C00-C16, or repeat --case for an
explicit subset. Every selected report must be bound to its prepared receipts,
actual adapter and training report, and the exact pre-evaluation problem map.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eesd.training_outcomes import tree_sha

ARMS = {
    "equal_uniform": "C01_EQUAL", "fixed_uniform": "C02_FIXED", "adaptive_uniform": "C03_EED",
    "fixed_edit": "C10_FIXED_EDIT", "adaptive_edit": "C11_EED_EDIT", "adaptive_random": "C12_EED_RANDOM_TOKEN",
}
P0_ARMS = {"frozen": "C00_FROZEN", **{k: v for k, v in ARMS.items() if v in
                                               ("C01_EQUAL", "C02_FIXED", "C03_EED")}}
CASE_ALIASES = {
    "C00_FROZEN": "frozen", "C01_EQUAL": "equal_uniform", "C02_FIXED": "fixed_uniform",
    "C03_EED": "adaptive_uniform", "C04_CONSTANT": "constant_mass", "C05_SHUFFLED": "shuffled_mass",
    "C06_NO_UNCERTAINTY": "no_uncertainty", "C07_FINAL_FILTER": "final_filter",
    "C08_CONFIDENCE": "confidence", "C09_TERMINAL_EED": "terminal_eed",
    "C10_FIXED_EDIT": "fixed_edit", "C11_EED_EDIT": "adaptive_edit",
    "C12_EED_RANDOM_TOKEN": "adaptive_random", "C13_KAPPA8": "kappa8", "C14_KAPPA32": "kappa32",
    "C15_UNIFORM_RELEVANCE": "uniform_relevance", "C16_PERMUTED_RELEVANCE": "permuted_relevance",
}
IDENTITY = ("model_id", "revision", "model_config_sha256", "shared_init_adapter_tree_sha256",
            "response_token_budget", "seed", "corrections_sha256", "trainer_source_sha256")
TRAIN_SETTINGS = ("max_steps", "gradient_accumulation", "max_length", "learning_rate", "optimizer", "lora_config")
ACTUAL_COMMON = ("training_schedule_schema", "training_schedule_sha256", "response_tokens",
                 "optimizer_steps", "micro_steps", "scheduled_training_trajectories", "loss_normalization")
SCHEDULE_SCHEMA = "eesd-scheduled-response-tokens-v1"
LEGACY_SCHEMA = "eesd-rebuttal-per-case-original-evaluator-v1"
LEGACY_PROTOCOL = "original-pbpf-greedy-sandbox-v1"
MATCHED_SCHEMA = "eesd-rebuttal-per-case-matched-evaluator-v1"
MATCHED_PROTOCOL = "eesd-rebuttal-matched-prospective-v1"
DEPENDENCY_FIELDS = ("dependency_versions", "generator_dependency_versions")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"object receipt required: {path}")
    return data


def require(data: dict, field: str, where: str):
    if field not in data or data[field] is None:
        raise ValueError(f"{where}: missing {field}")
    return data[field]


def same(actual, expected, where: str) -> None:
    if actual != expected:
        raise ValueError(f"{where}: {actual!r} != {expected!r}")


def file_bound(path: Path, expected: str, where: str) -> None:
    if not path.is_file() or not isinstance(expected, str) or sha(path) != expected:
        raise ValueError(f"{where}: file checksum mismatch")


def source_problem_map(path: Path) -> dict[str, str]:
    rows = [json.loads(x) for x in path.read_bytes().decode("utf-8").split("\n") if x.strip()]
    if not rows:
        raise ValueError("empty frozen source/problem map")
    mapping = {}
    for row in rows:
        sid, pid = row.get("source_component_id"), row.get("problem_id")
        if not isinstance(sid, str) or not sid or isinstance(pid, (bool, dict, list)) or pid is None or not str(pid):
            raise ValueError("source map must give real source_component_id and problem_id")
        if sid in mapping:
            raise ValueError("duplicate source mapping")
        mapping[sid] = str(pid)
    return mapping


def selected_arms(case_set: str, case_ids: list[str] | None) -> dict[str, str]:
    if case_ids is None:
        if case_set == "token":
            return dict(ARMS)
        if case_set == "p0":
            return dict(P0_ARMS)
        if case_set == "all":
            return {alias: case_id for case_id, alias in CASE_ALIASES.items()}
        raise ValueError("unknown case set")
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("explicit cases must be nonempty and unique")
    if any(case_id not in CASE_ALIASES for case_id in case_ids):
        raise ValueError("only registered C00-C16 baseline/training cases can be collected")
    return {CASE_ALIASES[case_id]: case_id for case_id in case_ids}


def _provenance(folder, case, plan, map_hash, mapping):
    path = folder / "data_provenance.json"
    file_bound(path, require(case, "data_provenance_sha256", "case provenance"), "case provenance")
    file_bound(path, require(plan, "data_provenance_sha256", "plan provenance"), "plan provenance")
    data = read_json(path)
    same(data.get("schema"), "eesd-token-aware-input-readiness-v1", "provenance schema")
    errors = require(data, "errors", "provenance audit errors")
    if not isinstance(errors, (dict, list)) or errors:
        raise ValueError(f"provenance audit errors: {errors!r}")
    same(data.get("provenance_status"), "problem_ids_disjoint_checked", "problem disjointness")
    same(plan.get("source_disjointness_status"), data["provenance_status"], "plan provenance status")
    if not isinstance(data.get("missing_problem_id_counts"), dict) or any(data["missing_problem_id_counts"].values()):
        raise ValueError("provenance has missing correction problem IDs")
    ev = require(data, "evaluation", "provenance")
    expected = {"eval_input_sha256": map_hash, "eval_rows": len(mapping),
                "eval_source_ids_supplied": len(mapping), "eval_problem_ids_supplied": len(set(mapping.values())),
                "source_level_overlap_checked": True, "problem_level_overlap_checked": True}
    for key, value in expected.items():
        same(ev.get(key), value, f"frozen problem map provenance {key}")
    same(case.get("frozen_case_input_hashes", {}).get("eval_ids_sha256"), map_hash, "frozen case problem map")
    same(plan.get("eval_source_id_file_sha256"), map_hash, "frozen evaluation problem map")
    file_bound(Path(require(plan, "eval_source_id_file_path", "frozen evaluation map")), map_hash, "frozen evaluation map")
    return data


def _bank_identity(plan, mapping):
    domain = require(plan, "domain", "evaluation source bank")
    replay = domain in ("apps_replay", "codecontests_replay")
    roots = {key: Path(require(plan, key, "evaluation source bank")).resolve()
             for key in ("public_root", "evaluator_root")}
    for root_key, prefix in (("public_root", "public"), ("evaluator_root", "evaluator")):
        folder = roots[root_key]
        for name in ("manifest", "tasks"):
            filename = "manifest.json" if name == "manifest" else "tasks.jsonl"
            file_bound(folder / filename, require(plan, f"{prefix}_{name}_sha256", "source bank"), "evaluation source bank")
        manifest = read_json(folder / "manifest.json")
        if replay:
            for key, value in {"schema": f"eesd-replay-{prefix}-v1", "domain": domain,
                               "task_kind": "stdin_synthesis"}.items():
                same(manifest.get(key), value, f"Replay source bank {key}")
            same(manifest.get("tasks", {}).get("sha256"), plan[f"{prefix}_tasks_sha256"], "Replay source bank task digest")
            same(manifest.get("tasks", {}).get("bytes"), (folder / "tasks.jsonl").stat().st_size, "Replay source bank task size")
        else:
            schema = {"rbr": "apbpf-rbr-generated-materialization-v1",
                      "codearc": "apbpf-codearc-replay-materialization-v1"}.get(domain)
            if schema is None:
                raise ValueError("unsupported evaluation source bank domain")
            same(manifest.get("schema"), schema, "source bank manifest schema")
            same(manifest.get(f"{prefix}_tasks_sha256"), plan[f"{prefix}_tasks_sha256"], "source bank task digest")
            same(manifest.get("public_tasks_sha256"), plan["public_tasks_sha256"], "public/private task origin")
        actual = {str(row["source_component_id"])
                  for line in (folder / "tasks.jsonl").read_bytes().decode("utf-8").split("\n")
                  if line.strip() for row in [json.loads(line)] if row.get("split") == "primary"}
        same(actual, set(mapping), "evaluation source bank primary inventory")
    if replay:
        public, evaluator = roots["public_root"], roots["evaluator_root"]
        if public.parent != evaluator.parent or public.parent.name != domain:
            raise ValueError("Replay public/evaluator roots must share their admitted domain directory")
        bundle = public.parent.parent
        admission_path = bundle / "admission.json"
        same(Path(require(plan, "admission_path", "Replay admission")).resolve(), admission_path,
             "Replay admission path")
        file_bound(admission_path, require(plan, "admission_sha256", "Replay admission"), "Replay admission")
        admission = read_json(admission_path)
        same(admission.get("schema"), "eesd-replay-admission-v1", "Replay admission schema")
        same(admission.get("status"), "data-admitted-not-generated-not-scored", "Replay admission status")
        for folder in (public, evaluator):
            for name in ("manifest.json", "tasks.jsonl"):
                path = folder / name
                locked = admission.get("outputs", {}).get(path.relative_to(bundle).as_posix(), {})
                same(locked.get("sha256"), sha(path), "Replay admission source digest")
                same(locked.get("bytes"), path.stat().st_size, "Replay admission source size")
    ids = require(plan, "evaluation_source_ids", "evaluation inventory")
    if not isinstance(ids, list) or len(ids) != len(set(ids)):
        raise ValueError("evaluation inventory must contain unique source IDs")
    same(set(ids), set(mapping), "evaluation source IDs")
    same(plan.get("evaluation_sources"), len(mapping), "evaluation source count")


def _training(folder, case, model, provenance):
    if case["case"]["kind"] == "baseline":
        return None, None
    setup = case["setup"]
    for key in (*IDENTITY, *TRAIN_SETTINGS, "beta", "alpha", "lambda_penalty", "token_edit_boost"):
        require(setup, key, "training binding setup")
    if type(setup["response_token_budget"]) is not int or setup["response_token_budget"] < 1:
        raise ValueError("training binding requires positive response token budget")
    same(setup["model_id"], model["model_id"], "training binding model")
    same(setup["revision"], model["revision"], "training binding revision")
    same(setup["seed"], case["seed"], "training binding seed")
    same(setup["corrections_sha256"], provenance.get("correction_file_sha256"), "training binding corrections provenance")
    correction_file = Path(require(setup, "corrections_path", "training binding"))
    file_bound(correction_file, setup["corrections_sha256"], "training binding correction bank")
    scored = folder / "scored-corrections.jsonl"
    scored_sha = require(case.get("frozen_case_input_hashes", {}), "scored-corrections.jsonl", "training binding input")
    file_bound(scored, scored_sha, "training binding scored input")
    path = folder / "model" / "training-report.json"
    file_bound(path, require(case, "training_report_sha256", "training binding receipt"), "training binding receipt")
    report = read_json(path)
    expected = {
        "schema": "eesd-weighted-sft-v1", "input_sha256": scored_sha, "rule": "eesd_full",
        "model_config_sha256": setup["model_config_sha256"], "model_id": model["model_id"],
        "revision": model["revision"], "seed": case["seed"],
        "response_tokens": setup["response_token_budget"], "response_token_budget": setup["response_token_budget"],
        "anchor_beta": setup["beta"], "include_zero_weight_kl": True,
        "token_weighting": case["case"]["token"], "token_edit_boost": setup["token_edit_boost"],
        "trainer_source_sha256": setup["trainer_source_sha256"],
        "previous_adapter_tree_sha256_before": setup["shared_init_adapter_tree_sha256"],
        "previous_adapter_tree_sha256_after": setup["shared_init_adapter_tree_sha256"],
        "training_schedule_schema": SCHEDULE_SCHEMA, "loss_normalization": "response_tokens",
        **{key: setup[key] for key in TRAIN_SETTINGS},
    }
    for key, value in expected.items():
        same(report.get(key), value, f"training binding {key}")
    schedule = require(report, "training_schedule_sha256", "training binding")
    if not isinstance(schedule, str) or not re.fullmatch(r"[0-9a-f]{64}", schedule):
        raise ValueError("training binding actual token schedule checksum missing or invalid")
    for key in ("optimizer_steps", "micro_steps", "scheduled_training_trajectories"):
        if type(report.get(key)) is not int or report[key] < 1:
            raise ValueError(f"training binding invalid actual {key}")
    common = {key: setup[key] for key in (*IDENTITY, *TRAIN_SETTINGS, "alpha", "lambda_penalty", "beta", "token_edit_boost")}
    common.update({key: report[key] for key in ACTUAL_COMMON})
    # Software receipts are retained/comparable, without requiring the current working
    # tree to still be the historical source revision used for training.
    for key in ("source_sha256", "scorer_source_sha256", "token_source_sha256", "registry_sha256"):
        if key in setup:
            common[key] = setup[key]
    if "dependency_versions" in report:
        versions = report["dependency_versions"]
        if not isinstance(versions, dict) or not versions:
            raise ValueError("training binding dependency_versions must be a nonempty version map")
        common["dependency_versions"] = versions
    detail = {"report_sha256": sha(path), "scored_input_sha256": scored_sha,
              "token_weighting": report["token_weighting"], **{key: report[key] for key in ACTUAL_COMMON}}
    if "dependency_versions" in common:
        detail["dependency_versions"] = common["dependency_versions"]
    return common, detail


def verify_records(report: dict, mapping: dict[str, str]) -> dict[str, tuple[str, ...]]:
    rows = report.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("report binding: no raw evaluator records")
    ids = [row.get("source_component_id") for row in rows]
    if any(not isinstance(sid, str) for sid in ids) or len(ids) != len(set(ids)) or set(ids) != set(mapping):
        raise ValueError("report binding: missing/duplicate/extra source IDs")
    inventory = {}
    for row in rows:
        sid = row["source_component_id"]
        if not isinstance(row.get("all_tests_pass"), bool):
            raise ValueError("nonboolean all-tests success")
        tests = row.get("tests")
        if not isinstance(tests, list) or not tests:
            raise ValueError("raw test outcomes are required")
        tids = []
        for test in tests:
            if not isinstance(test, dict) or not isinstance(test.get("outcome"), str) or not test["outcome"]:
                raise ValueError("invalid raw test outcomes")
            tid = test.get("test_id")
            if isinstance(tid, (bool, dict, list)) or tid is None or str(tid) == "":
                raise ValueError("raw test outcomes require test IDs")
            tids.append(str(tid))
        if len(set(tids)) != len(tids):
            raise ValueError("duplicate raw test outcomes")
        same(row["all_tests_pass"], all(t["outcome"] == "PASS" for t in tests), "all-tests success contradicts raw test outcomes")
        if "problem_id" in row:
            same(str(row["problem_id"]), mapping[sid], "frozen problem mapping contradicts evaluator row")
        row["problem_id"] = mapping[sid]
        inventory[sid] = tuple(sorted(tids))
    same(report.get("sources"), len(rows), "report binding source count")
    passes = sum(row["all_tests_pass"] for row in rows)
    if "all_tests_passes" in report:
        same(report["all_tests_passes"], passes, "aggregate test outcomes count")
    if "fresh_all_tests_pass_at_1" in report:
        value = report["fresh_all_tests_pass_at_1"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not math.isclose(value, passes / len(rows), abs_tol=1e-12):
            raise ValueError("aggregate test outcomes success fraction differs")
    return inventory


def collect(root: Path, study: str, seed: int, problem_map: Path, output: Path, *,
            case_set: str = "token", case_ids: list[str] | None = None,
            evaluation_subdir: str = "evaluation") -> dict:
    """Collect a selected set after all verification succeeds; old five args remain valid."""
    if output.exists():
        raise FileExistsError("collection already exists")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", study) or evaluation_subdir not in ("evaluation", "matched_evaluation", "evaluation_matched", "matched-evaluation"):
        raise ValueError("invalid study or evaluation directory")
    arms = selected_arms(case_set, case_ids)
    registry = read_json(ROOT / "configs/rebuttal/eesd_case_registry.json")
    registered = {case["id"]: case for case in registry["cases"]}
    mapping, source_map_sha = source_problem_map(problem_map), sha(problem_map)
    cases, raw = {}, {}
    first_common = first_eval = first_initial = first_tests = first_provenance = None
    for alias, case_id in arms.items():
        folder = root / study / case_id / f"seed{seed}"
        receipt, ep = folder / "case.json", folder / evaluation_subdir / "plan.json"
        result = ep.parent / "heldout" / "report.json"
        if not all(path.is_file() for path in (receipt, ep, result)):
            raise FileNotFoundError(f"{case_id}: case/evaluator receipt missing")
        case, plan, report = read_json(receipt), read_json(ep), read_json(result)
        same(case.get("schema"), "eesd-iclr2027-independent-case-v1", "case receipt schema")
        same(case.get("study"), study, "case binding study")
        same(case.get("seed"), seed, "case binding seed")
        spec = require(case, "case", "case binding")
        for key in ("id", "kind", "weight", "token", "kappa", "relevance_mode"):
            same(spec.get(key), registered[case_id].get(key), f"case binding {key}")
        baseline = spec["kind"] == "baseline"
        same(case.get("status"), "completed_diagnostic_without_gpu" if baseline else "trained_awaiting_independent_evaluation", "case receipt status")
        same(plan.get("status"), "finished_with_real_heldout", "evaluation plan status")
        for key, value in {"case_sha256": sha(receipt), "case_id": case_id, "study": study, "seed": seed}.items():
            same(plan.get(key), value, f"case binding plan receipt {key}")
        file_bound(result, plan.get("heldout_report_sha256"), "heldout report binding")
        protocol = require(plan, "evaluation_protocol", "evaluation protocol")
        if plan.get("schema") == LEGACY_SCHEMA:
            same(protocol, LEGACY_PROTOCOL, "evaluation protocol schema")
        elif plan.get("schema") == MATCHED_SCHEMA:
            same(protocol, MATCHED_PROTOCOL, "evaluation protocol schema")
        else:
            raise ValueError("unsupported evaluation protocol plan; only sealed original or prospective matched plans are accepted")
        setup = require(case, "setup", "case binding")
        model_path = Path(require(setup, "model_config_path", "model binding"))
        file_bound(model_path, require(setup, "model_config_sha256", "model binding"), "model binding")
        same(plan.get("model_config_sha256"), sha(model_path), "model binding plan")
        file_bound(Path(require(plan, "model_config_path", "model binding plan")), sha(model_path), "model binding plan")
        cfg = yaml.safe_load(model_path.read_text())
        if not isinstance(cfg, dict) or not isinstance(cfg.get("model_id"), str) or not re.fullmatch(r"[0-9a-f]{40}", str(cfg.get("revision", ""))):
            raise ValueError("model binding requires a pinned model revision")
        for key in ("model_id", "revision"):
            same(plan.get(key), cfg[key], f"model binding plan {key}")
            if key in setup:
                same(setup[key], cfg[key], f"model binding case {key}")
        provenance = _provenance(folder, case, plan, source_map_sha, mapping)
        provenance_id = require(provenance, "correction_file_sha256", "correction provenance")
        if first_provenance is None:
            first_provenance = provenance_id
        same(provenance_id, first_provenance, "correction provenance differs across cases")
        initial_path = Path(require(setup, "shared_init_adapter_path", "initial adapter binding"))
        initial = tree_sha(initial_path)
        same(initial, setup.get("shared_init_adapter_tree_sha256"), "initial adapter binding")
        if first_initial is None:
            first_initial = initial
        same(initial, first_initial, "initial adapter differs across cases")
        expected_adapter = initial_path if baseline else folder / "model" / "adapter"
        adapter_hash = tree_sha(expected_adapter)
        if not baseline:
            same(case.get("trained_adapter_tree_sha256"), adapter_hash, "trained adapter binding")
        same(plan.get("adapter_tree_sha256"), adapter_hash, "evaluated adapter binding")
        same(Path(require(plan, "adapter_path", "evaluated adapter binding")).resolve(), expected_adapter.resolve(), "evaluated adapter path")
        _bank_identity(plan, mapping)
        same(report.get("schema"), "eesd-fresh-policy-eval-v1", "report binding schema")
        for key, value in {"model": cfg["model_id"], "revision": cfg["revision"], "domain": require(plan, "domain", "evaluation protocol"),
                           "adapter_sha256": adapter_hash, "task_manifest_sha256": plan["evaluator_manifest_sha256"],
                           "execution_profile": "sandbox", "sources": len(mapping)}.items():
            same(report.get(key), value, f"report binding {key}")
        if plan["schema"] == MATCHED_SCHEMA:
            same(require(report, "evaluation_protocol", "report binding"), protocol, "report binding evaluation protocol")
            policy = require(plan, "generation_policy", "generation policy")
            if not isinstance(policy, dict):
                raise ValueError("generation policy must be an object")
            same(require(report, "generation_policy", "report binding"), policy, "report binding generation policy")
        elif "generation_policy" in report and "generation_policy" in plan:
            same(report["generation_policy"], plan["generation_policy"], "report binding generation policy")
        for key in DEPENDENCY_FIELDS:
            if key in report:
                if not isinstance(report[key], dict) or not report[key]:
                    raise ValueError(f"report binding {key} must be a nonempty version map")
                if key in plan:
                    same(report[key], plan[key], f"report binding {key}")
        if "seed" in report:
            same(report["seed"], seed, "report binding seed")
        if "evaluation_protocol" in report:
            same(report["evaluation_protocol"], protocol, "report binding evaluation protocol")
        if case.get("original_paper_cell") != plan.get("original_paper_cell"):
            raise ValueError("case binding original paper cell differs")
        tests = verify_records(report, mapping)
        if first_tests is None:
            first_tests = tests
        same(tests, first_tests, "evaluation protocol test inventory differs")
        for key, value in (("heldout_count", len(mapping)), ("heldout_sources", len(mapping)), ("heldout_all_tests_pass_at_1", sum(r["all_tests_pass"] for r in report["records"]) / len(mapping))):
            if key in plan:
                same(plan[key], value, f"report binding plan {key}")
        identity = {key: report.get(key) for key in ("model", "revision", "domain", "task_manifest_sha256", "execution_profile", "execution_lock_sha256", "sources")}
        identity.update({"evaluation_protocol": protocol, "model_config_sha256": sha(model_path),
                         "shared_init_adapter_tree_sha256": initial, "problem_map_sha256": source_map_sha})
        for key in ("pbpf_commit", "public_manifest_sha256", "public_tasks_sha256", "evaluator_tasks_sha256",
                    "generator_sha256", "evaluator_sha256", "evaluation_source_sha256", "protocol_config_sha256",
                    "generation_policy", "execution_policy", "quantization", "original_paper_cell", "admission_sha256"):
            if key in plan:
                identity[key] = plan[key]
        for key in DEPENDENCY_FIELDS:
            if key in report:
                identity[key] = report[key]
        if plan["schema"] == MATCHED_SCHEMA:
            for key in ("local_source_hashes", "pbpf_source_hashes"):
                hashes = require(plan, key, "matched evaluator source binding")
                if not isinstance(hashes, dict) or not hashes or any(
                    not isinstance(name, str) or not name
                    or not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                    for name, value in hashes.items()
                ):
                    raise ValueError(f"matched evaluator source binding: invalid {key}")
                # Compare the frozen implementations; historical runs need not
                # use the collector's current checkout.
                identity[key] = hashes
        if first_eval is None:
            first_eval = identity
        same(identity, first_eval, "evaluation protocol or population differs")
        common, training = _training(folder, case, cfg, provenance)
        if common is not None:
            if first_common is None:
                first_common = common
            for key in common.keys() | first_common.keys():
                same(common.get(key), first_common.get(key), f"training settings/schedule differ in {key}")
        cases[alias] = {"case_id": case_id, "case_specification": spec, "case_sha256": sha(receipt),
                        "evaluation_plan_sha256": sha(ep), "raw_evaluation_report_sha256": sha(result),
                        "adapter_tree_sha256": adapter_hash, "data_provenance_sha256": sha(folder / "data_provenance.json"),
                        "training": training, "report_file": alias + ".json"}
        raw[alias] = report
    # Complete all checks and serialization before creating the output directory.
    serialized = {alias: json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n" for alias, value in raw.items()}
    for alias, text in serialized.items():
        cases[alias]["collected_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    legacy = arms == ARMS
    manifest = {
        "schema": "eesd-iclr2027-six-arm-heldout-collection-v1" if legacy else "eesd-iclr2027-case-heldout-collection-v1",
        "status": "six_real_heldout_reports_verified_not_statistically_interpreted" if legacy else "selected_real_heldout_reports_verified_not_statistically_interpreted",
        "study": study, "seed": seed, "case_ids": list(arms.values()), "problem_map_sha256": source_map_sha,
        "problem_map_file": "problem-map.jsonl",
        "source_count": len(mapping), "distinct_problem_count": len(set(mapping.values())),
        "evaluation_identity": first_eval, "common_training_parameters": first_common,
        "corrections_sha256": first_provenance, "arms": cases,
        "limitation": "Checks bind preserved local receipts and files, not an independently signed dataset origin. The exact pre-evaluation problem map and audited correction-bank identity are required. Sampling uncertainty is conditional on these fitted models; no new outcomes are generated.",
    }
    manifest_text = json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    map_bytes = problem_map.read_bytes()
    same(hashlib.sha256(map_bytes).hexdigest(), source_map_sha, "frozen map changed while collecting")
    output.mkdir(parents=True, exist_ok=False)
    (output / "problem-map.jsonl").write_bytes(map_bytes)
    for alias, text in serialized.items():
        (output / (alias + ".json")).write_text(text)
    (output / "manifest.json").write_text(manifest_text)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-root", type=Path, default=Path("runs/eesd-iclr2027-rebuttal"))
    parser.add_argument("--study", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--problem-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--case-set", "--suite", dest="case_set", choices=("token", "p0", "all"), default="token")
    group.add_argument("--case", dest="case_ids", action="append", choices=tuple(CASE_ALIASES))
    parser.add_argument("--evaluation-subdir", default="evaluation")
    args = parser.parse_args()
    result = collect(args.cases_root, args.study, args.seed, args.problem_map, args.output,
                     case_set=args.case_set, case_ids=args.case_ids, evaluation_subdir=args.evaluation_subdir)
    print(json.dumps({key: result[key] for key in ("status", "study", "case_ids", "source_count", "distinct_problem_count")}, indent=2))


if __name__ == "__main__":
    main()
