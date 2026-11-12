#!/usr/bin/env python3
"""Create exactly one shared Qwen2.5-Coder-7B LoRA initialization for all arms.

Do this ONCE on an isolated H100 host with cached pinned model weights. Never
allow independent arms to invent an initial LoRA initialization silently.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_recursive_shapley_sdpo as legacy


def digest_tree(root:Path)->str:
    paths=sorted(p for p in root.rglob("*") if p.is_file())
    if not paths:raise ValueError('no checkpoint files to seal')
    h=hashlib.sha256()
    for p in paths:
        name=p.relative_to(root).as_posix().encode()
        h.update(len(name).to_bytes(4,'big'));h.update(name)
        with p.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=1701)
    p.add_argument('--model-config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/models/qwen25_7b.yaml')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('refuse to overwrite shared model initialization')
    legacy._require_model_config(a.model_config)
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 CUDA required for real Qwen initialization')
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    args=type('Args',(),{'model_config':a.model_config})()
    _,_,attach=legacy._models(args)
    model=attach(None,trainable=False)
    a.output.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(a.output)
    receipt={'schema':'eesd-shared-initial-lora-v1','seed':a.seed,
             'model_id':legacy.MODEL_ID,'revision':legacy.MODEL_REVISION,
             'sha256':digest_tree(a.output)}
    (a.output.parent/(a.output.name+'.receipt.json')).write_text(
          json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
