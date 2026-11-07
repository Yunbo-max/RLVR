"""CPU regressions for immutable training identity; no model results are produced."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from rebuttal_test_utils import write_lora_adapter
from eesd.training_outcomes import adapter_metadata, gemma_text_lora_targets
from test_eesd_rebuttal_independent_cases import fixture_paths, m
from run_eesd_weighted_sft import SCHEDULE_SCHEMA, lora_parameters, schedule_step_bytes


def prepared(tmp_path):
    args = fixture_paths(tmp_path)
    case = m.load_cases()["C03_EED"]
    folder = tmp_path / "case"
    manifest = m.prepare_case(args, case, folder)
    return args, case, folder, manifest


def matching_report(manifest):
    setup, case = manifest["setup"], manifest["case"]
    return {
        "schema": "eesd-weighted-sft-v1",
        "input_sha256": manifest["frozen_case_input_hashes"]["scored-corrections.jsonl"],
        "model_config_sha256": setup["model_config_sha256"],
        "model_id": setup["model_id"], "revision": setup["revision"],
        "seed": setup["seed"], "rule": "eesd_full",
        "response_tokens": setup["response_token_budget"],
        "response_token_budget": setup["response_token_budget"],
        "anchor_beta": setup["beta"], "include_zero_weight_kl": True,
        "token_weighting": case["token"], "token_edit_boost": setup["token_edit_boost"],
        "trainer_source_sha256": setup["trainer_source_sha256"],
        "previous_adapter_tree_sha256_before": setup["shared_init_adapter_tree_sha256"],
        "previous_adapter_tree_sha256_after": setup["shared_init_adapter_tree_sha256"],
        "optimizer": copy.deepcopy(setup["optimizer"]),
        "lora_config": copy.deepcopy(setup["lora_config"]),
        "training_schedule_schema": SCHEDULE_SCHEMA,
        "training_schedule_sha256": "1" * 64,
        **{key: setup[key] for key in m.TRAINING_DEFAULTS},
    }


@pytest.mark.parametrize("damage", ["wrong_model", "wrong_revision", "missing_weights", "truncated_weights"])
def test_prepare_rejects_invalid_adapter_before_creating_case(tmp_path, damage):
    args = fixture_paths(tmp_path)
    config_path = args.init_adapter / "adapter_config.json"
    cfg = json.loads(config_path.read_text())
    if damage in ("wrong_model", "wrong_revision"):
        cfg["base_model_name_or_path" if damage == "wrong_model" else "revision"] = "unrelated"
        config_path.write_text(json.dumps(cfg))
    else:
        weights = args.init_adapter / "adapter_model.safetensors"
        if damage == "missing_weights":
            weights.unlink()
        else:
            weights.write_bytes(weights.read_bytes()[:-1])
    folder = tmp_path / "case"
    with pytest.raises(ValueError, match="invalid parent adapter"):
        m.prepare_case(args, m.load_cases()["C03_EED"], folder)
    assert not folder.exists()


@pytest.mark.parametrize("flag,value", [
    ("--seed", "7"), ("--learning-rate", "0.01"),
    ("--rule", "equal_weight"), ("--previous-adapter", "/unrelated/adapter"),
])
def test_command_mutation_fails_before_any_subprocess(tmp_path, monkeypatch, flag, value):
    _, _, folder, manifest = prepared(tmp_path)
    manifest["command"][manifest["command"].index(flag) + 1] = value
    (folder / "case.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: pytest.fail("GPU subprocess must not start"))
    with pytest.raises(ValueError, match="frozen trainer command"):
        m.run_case(folder)
    assert json.loads((folder / "case.json").read_text())["status"] == "prepared"


def test_registry_specification_mutation_is_rejected(tmp_path):
    _, _, folder, manifest = prepared(tmp_path)
    manifest["case"]["weight"] = "equal"
    (folder / "case.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="frozen registry"):
        m.run_case(folder)


def test_scoring_dependency_change_invalidates_prepared_case(tmp_path, monkeypatch):
    _, _, _, manifest = prepared(tmp_path)
    real_sha = m.file_sha
    changed = m.ROOT / "src/eesd/evidence.py"
    monkeypatch.setattr(m, "file_sha", lambda path: "0" * 64 if path == changed else real_sha(path))
    with pytest.raises(ValueError, match="evidence.py"):
        m.check_source_integrity(manifest["setup"])


@pytest.mark.parametrize("field,value", [
    ("previous_adapter_tree_sha256_after", "2" * 64),
    ("learning_rate", 0.01), ("max_length", 1024),
    ("training_schedule_sha256", None), ("optimizer", {}), ("lora_config", {}),
])
def test_training_receipt_must_bind_actual_run_to_setup(tmp_path, field, value):
    _, case, _, manifest = prepared(tmp_path)
    report = matching_report(manifest)
    m.verify_training_report(report, manifest["setup"], case, report["input_sha256"])
    report[field] = value
    with pytest.raises(ValueError, match="TRAINING BINDING ERROR"):
        m.verify_training_report(report, manifest["setup"], case,
                                 manifest["frozen_case_input_hashes"]["scored-corrections.jsonl"])


def test_cpu_mock_completion_preserves_adapter_and_records_verified_identity(tmp_path, monkeypatch):
    args, _, folder, manifest = prepared(tmp_path)
    initial_hash = m.tree_sha(args.init_adapter)

    def synthetic_subprocess(command, **kwargs):
        # Deliberate test double: this does not load a model or simulate results.
        write_lora_adapter(folder / "model" / "adapter", args.model_config)
        (folder / "model" / "training-report.json").write_text(json.dumps(matching_report(manifest)))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(m.subprocess, "run", synthetic_subprocess)
    result = m.run_case(folder)
    assert result["status"] == "trained_awaiting_independent_evaluation"
    assert m.tree_sha(args.init_adapter) == initial_hash
    assert result["training_report_sha256"] == m.file_sha(folder / "model" / "training-report.json")


def test_schedule_hash_excludes_arm_weights_but_binds_order_and_token_prefix():
    row = {"trajectory_id": "t1", "source_component_id": "s1", "_weight": 0.1}
    encoded = {"prompt_ids": [11, 12], "completion_ids": [21, 22, 23]}
    first = schedule_step_bytes(row, encoded, 2)
    assert first == schedule_step_bytes({**row, "_weight": 9.0}, encoded, 2)
    assert first != schedule_step_bytes(row, encoded, 3)
    second = schedule_step_bytes({**row, "trajectory_id": "t2"}, encoded, 2)
    assert hashlib.sha256(first + second).hexdigest() != hashlib.sha256(second + first).hexdigest()
    assert first != schedule_step_bytes(row, {**encoded, "prompt_ids": [99]}, 2)
    with pytest.raises(ValueError, match="scheduled response prefix"):
        schedule_step_bytes(row, encoded, 4)


def test_loaded_lora_parameters_have_stable_order_and_retain_actual_values():
    cfg = {"r": 8, "lora_alpha": 7.5, "lora_dropout": 0.1,
           "target_modules": {"v_proj", "q_proj"}, "peft_type": "LORA", "task_type": "CAUSAL_LM"}
    result = lora_parameters(cfg)
    assert (result["r"], result["lora_alpha"], result["lora_dropout"]) == (8, 7.5, 0.1)
    assert result == lora_parameters({**cfg, "target_modules": ["q_proj", "v_proj"]})


@pytest.mark.parametrize("key,value", [
    ("revision", None), ("target_modules", None), ("target_modules", []),
    ("target_modules", ""), ("target_modules", [None]),
])
def test_prepare_requires_explicit_revision_and_adapter_targets(tmp_path, key, value):
    args = fixture_paths(tmp_path)
    config_path = args.init_adapter / "adapter_config.json"
    cfg = json.loads(config_path.read_text())
    cfg[key] = value
    config_path.write_text(json.dumps(cfg))
    folder = tmp_path / "case"
    with pytest.raises(ValueError, match="invalid parent adapter"):
        m.prepare_case(args, m.load_cases()["C03_EED"], folder)
    assert not folder.exists()


@pytest.mark.parametrize("prefix", ["language_model.model.layers.", "model.language_model.layers."])
def test_gemma_targets_support_both_text_layouts_and_exclude_vision(prefix):
    text_targets = [prefix + "0.self_attn.q_proj", prefix + "0.mlp.down_proj"]
    other = ["model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
             "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
             prefix + "0.self_attn.not_q_proj", prefix + "0.input_layernorm",
             "prefix." + text_targets[0]]
    assert gemma_text_lora_targets(iter(text_targets + other)) == sorted(text_targets)
    with pytest.raises(ValueError, match="text decoder LoRA targets not found"):
        gemma_text_lora_targets(other)


@pytest.mark.parametrize("targets", [
    "all-linear", ["q_proj", "v_proj"],
    ["model.language_model.layers.0.self_attn.q_proj", "model.vision_tower.q_proj"],
])
def test_gemma_adapter_requires_text_only_targets(tmp_path, targets):
    model_config = m.ROOT / "configs/rebuttal/gemma3_4b.yaml"
    directory = tmp_path / "adapter"
    cfg = write_lora_adapter(directory, model_config)
    assert adapter_metadata(directory, model_config) == cfg
    cfg["target_modules"] = targets
    (directory / "adapter_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="invalid parent adapter"):
        adapter_metadata(directory, model_config)
