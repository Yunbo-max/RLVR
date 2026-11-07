"""Small CPU-only fixtures for checkpoint identity checks; never trainable models."""
import json
import struct
from pathlib import Path

import yaml


def write_lora_adapter(directory, model_config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model = yaml.safe_load(Path(model_config).read_text())
    config = {"peft_type": "LORA", "task_type": "CAUSAL_LM",
              "base_model_name_or_path": model["model_id"],
              "revision": model["revision"], "r": 16, "lora_alpha": 32,
              "lora_dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}
    if model["model_id"] == "google/gemma-3-4b-it":
        config["target_modules"] = ["model.language_model.layers.0.self_attn.q_proj"]
    (directory / "adapter_config.json").write_text(json.dumps(config))
    header = {
        "base_model.model.layers.0.q_proj.lora_A.weight": {
            "dtype": "F32", "shape": [16, 1], "data_offsets": [0, 64]},
        "base_model.model.layers.0.q_proj.lora_B.weight": {
            "dtype": "F32", "shape": [1, 16], "data_offsets": [64, 128]},
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    (directory / "adapter_model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"\0" * 128)
    return config
