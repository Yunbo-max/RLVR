#!/usr/bin/env python3
"""Create ONE pinned, zero-delta LoRA initialization shared by a rebuttal cell.

Supports the original Qwen, DeepSeek and Gemma backbones. Run once per model
and fit seed on a BF16 CUDA host with locally cached weights. All cases,
including C00, must reuse the resulting adapter. --dry-run needs no ML stack.
The receipt is a sibling of the adapter, so its checksum is not self-referential.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eesd.training_outcomes import tree_sha, gemma_text_lora_targets
from check_eesd_rebuttal_coverage import load_matrix


def initialization_spec(model_config: Path, *, seed: int) -> dict:
    if type(seed) is not int or seed < 0:
        raise ValueError("initialization seed must be a nonnegative integer")
    cfg = yaml.safe_load(model_config.read_text())
    pins = {(v["model_id"], v["model_revision"]) for v in load_matrix().values()}
    if not isinstance(cfg, dict) or (cfg.get("model_id"), cfg.get("revision")) not in pins:
        raise ValueError("model config must match an original pinned rebuttal backbone")
    return {
        "schema": "eesd-rebuttal-shared-init-v1",
        "status": "planned_not_created",
        "model_id": cfg["model_id"],
        "revision": cfg["revision"],
        "model_config_sha256": hashlib.sha256(model_config.read_bytes()).hexdigest(),
        "seed": seed,
        "lora": {"r": 16, "alpha": 32, "dropout": 0.0, "bias": "none", "task_type": "CAUSAL_LM"},
        "quantization": "nf4-double-quant-bfloat16",
        "base_weights": "local_files_only",
        "initial_policy": "zero LoRA B matrices; no optimizer update",
        "scientific_status": "initialization only, not training or heldout evaluation",
    }


def lora_targets(model_id: str, module_names) -> list[str] | str:
    if model_id != "google/gemma-3-4b-it":
        return "all-linear"
    return gemma_text_lora_targets(module_names)


def create(output: Path, spec: dict) -> dict:
    receipt_path = output.with_name(output.name + ".receipt.json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError("shared initialization or receipt already exists; choose a new output")
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("shared rebuttal initialization requires BF16-capable CUDA")
    random.seed(spec["seed"])
    torch.manual_seed(spec["seed"])
    torch.cuda.manual_seed_all(spec["seed"])
    model_class = AutoModelForCausalLM
    if spec["model_id"] == "google/gemma-3-4b-it":
        from transformers import Gemma3ForConditionalGeneration
        model_class = Gemma3ForConditionalGeneration
    base = model_class.from_pretrained(
        spec["model_id"], revision=spec["revision"], local_files_only=True,
        device_map={"": 0}, torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        ),
    )
    base = prepare_model_for_kbit_training(base)
    targets = lora_targets(spec["model_id"], (name for name, _ in base.named_modules()))
    lora = spec["lora"]
    model = get_peft_model(base, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        bias=lora["bias"], task_type=lora["task_type"], target_modules=targets,
        revision=spec["revision"], init_lora_weights=True,
    ))
    model.peft_config["default"].revision = spec["revision"]
    b_parameters = [p for name, p in model.named_parameters() if ".lora_B." in name]
    if not b_parameters or any(torch.count_nonzero(p).item() != 0 for p in b_parameters):
        raise ValueError("initial adapter must have exactly zero LoRA B matrices")
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output, safe_serialization=True)
    if not (output / "adapter_config.json").is_file() or not (output / "adapter_model.safetensors").is_file():
        raise RuntimeError("initialization did not publish an adapter config and weights")
    receipt = {
        **spec,
        "status": "created_zero_delta_shared_initialization",
        "adapter_path": str(output.resolve()),
        "adapter_tree_sha256": tree_sha(output),
        "lora_target_modules": targets,
        "zero_lora_B_matrices": len(b_parameters),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "bitsandbytes")},
    }
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_name(args.output.name + ".receipt.json").exists():
        raise FileExistsError("refuse to overwrite shared initialization or receipt")
    spec = initialization_spec(args.model_config, seed=args.seed)
    print(json.dumps(spec if args.dry_run else create(args.output, spec), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
