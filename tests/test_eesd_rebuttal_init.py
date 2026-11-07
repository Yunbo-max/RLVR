"""The three rebuttal backbones need reproducible, text-only shared starts."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/create_eesd_rebuttal_init.py"


def module():
    assert SCRIPT.is_file(), "there is no shared initializer for DeepSeek/Gemma rebuttal cases"
    spec = importlib.util.spec_from_file_location("rebuttal_init", SCRIPT)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.mark.parametrize("config,model", [
    ("qwen25_coder_7b.yaml", "Qwen/Qwen2.5-Coder-7B-Instruct"),
    ("deepseek_coder_6p7b.yaml", "deepseek-ai/deepseek-coder-6.7b-instruct"),
    ("gemma3_4b.yaml", "google/gemma-3-4b-it"),
])
def test_each_model_can_plan_without_gpu_or_overwriting(tmp_path, config, model):
    module()
    destination = tmp_path / "shared"
    run = subprocess.run([
        sys.executable, str(SCRIPT), "--model-config", str(ROOT / "configs/rebuttal" / config),
        "--seed", "1702", "--output", str(destination), "--dry-run",
    ], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    spec = json.loads(run.stdout)
    assert spec["model_id"] == model and len(spec["revision"]) == 40
    assert spec["seed"] == 1702
    assert spec["lora"] == {"r": 16, "alpha": 32, "dropout": 0.0, "bias": "none", "task_type": "CAUSAL_LM"}
    assert spec["quantization"] == "nf4-double-quant-bfloat16"
    assert spec["status"] == "planned_not_created"
    assert not destination.exists()
    destination.mkdir()
    (destination / "keep.txt").write_text("keep")
    again = subprocess.run(run.args, capture_output=True, text=True)
    assert again.returncode != 0 and (destination / "keep.txt").read_text() == "keep"


def test_gemma_targets_exclude_vision_and_embedding():
    m = module()
    names = [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.mlp.up_proj",
        "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
        "model.multi_modal_projector.linear",
        "model.language_model.embed_tokens",
    ]
    assert m.lora_targets("google/gemma-3-4b-it", names) == sorted(names[:2])
    with pytest.raises(ValueError, match="text decoder"):
        m.lora_targets("google/gemma-3-4b-it", names[2:])


def test_gemma_451_text_namespace_is_supported_without_vision():
    m = module()
    names = [
        "language_model.model.layers.0.self_attn.q_proj",
        "language_model.model.layers.0.mlp.down_proj",
        "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
        "language_model.lm_head",
    ]
    assert m.lora_targets("google/gemma-3-4b-it", names) == sorted(names[:2])


def test_rejects_a_changed_model_pin_or_negative_seed(tmp_path):
    m = module()
    cfg = tmp_path / "model.yaml"
    cfg.write_text("model_id: google/gemma-3-4b-it\nrevision: " + "a" * 40 + "\n")
    with pytest.raises(ValueError, match="pinned"):
        m.initialization_spec(cfg, seed=1701)
    with pytest.raises(ValueError, match="seed"):
        m.initialization_spec(ROOT / "configs/rebuttal/gemma3_4b.yaml", seed=-1)
