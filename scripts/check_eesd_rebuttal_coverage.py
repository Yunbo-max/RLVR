#!/usr/bin/env python3
"""Coverage and identity gate for all twelve ORIGINAL ICLR2027 EED model/data cells.

Treat registered training support as code coverage, NOT as evidence that raw
corrections, H100 model weights, or a valid evaluator are available. All
twelve cells have a model-specific LoRA trainer; only Qwen+DeepSeek on
RunBugRun+CodeARC have a pinned PBPF sandbox generation+evaluation bridge.
Gemma and Replay have historical, nonportable evaluators; no fictitious
"compatible" protocol is substituted.

For each cell optional frozen inputs:
  <INPUTS_ROOT>/<cell>/corrections.jsonl
  <INPUTS_ROOT>/<cell>/eval_source_problem_ids.jsonl
  <INPUTS_ROOT>/<cell>/shared_initial_adapter/adapter_config.json
An audit never downloads, executes or modifies source data.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping

import yaml

ROOT=Path(__file__).resolve().parents[1]
MATRIX=ROOT/"configs/rebuttal/model_dataset_matrix.json"
DATASETS=("runbugrun","codearc","apps_replay","codecontests_replay")
MODELS=("qwen25_7b","deepseek_6p7b","gemma3_4b")
ALIASES={"rbr":"runbugrun","runbugrun":"runbugrun",
         "codearc":"codearc","apps_replay":"apps_replay",
         "codecontests_replay":"codecontests_replay"}


def hash_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def load_matrix()->dict[str,dict]:
    data=json.loads(MATRIX.read_text())
    if data.get("schema")!="eesd-iclr2027-model-dataset-coverage-v1":
        raise ValueError("invalid coverage registry")
    if data.get("dataset_keys")!=list(DATASETS) or data.get("model_keys")!=list(MODELS):
        raise ValueError("model/domain family registry changed without study revision")
    cells={}
    for entry in data.get("cells",[]):
        name=entry["cell"]
        if name in cells or name!=f'{entry["dataset"]}__{entry["model"]}':
            raise ValueError("duplicate or malformed cell ID")
        if entry["dataset"] not in DATASETS or entry["model"] not in MODELS:
            raise ValueError("unknown original EED dataset/model")
        model_path=ROOT/entry["model_config"]
        if not model_path.is_file():
            raise FileNotFoundError(f"missing model YAML {model_path}")
        cfg=yaml.safe_load(model_path.read_text())
        if (cfg.get("model_id")!=entry["model_id"]
            or not isinstance(cfg.get("revision"),str)
            or len(cfg["revision"])!=40):
            raise ValueError(f"model revision mismatch for {name}")
        cells[name]={**entry,"model_revision":cfg["revision"],
                     "model_config_sha256":hash_file(model_path)}
    if set(cells)!={f"{d}__{m}" for d in DATASETS for m in MODELS}:
        raise ValueError("matrix must contain exactly 3x4 original paper cells")
    return cells


def audit_correction_identity(cell:str,rows_path:Path,*,strict:bool=False)->dict:
    cells=load_matrix()
    if cell not in cells:raise ValueError(f"unknown model/dataset cell {cell!r}")
    profile=cells[cell]
    if not rows_path.is_file():raise FileNotFoundError(rows_path)
    declared=Counter()
    missing=Counter()
    samples=0
    for line in rows_path.read_text().split("\n"):
        if not line.strip():continue
        row=json.loads(line)
        if not isinstance(row,dict):raise ValueError("non-object correction row")
        samples+=1
        # Identify before/after correction banks by their original archived fields.
        for key in ("model_id","model_revision"):
            if key not in row:
                missing[key]+=1
            else:
                expected=profile["model_id"] if key=="model_id" else profile["model_revision"]
                if row[key]!=expected:
                    raise ValueError(f"{cell}: correction {key} is not the selected frozen backbone")
                declared[key]+=1
        explicit=[]
        for field in ("dataset","source_dataset","domain"):
            if field not in row:continue
            dataset=ALIASES.get(row[field])
            if dataset is None or dataset!=profile["dataset"]:
                raise ValueError(f"{cell}: correction {field} is not the selected dataset")
            explicit.append(field)
            declared[field]+=1
        if not explicit:
            missing["dataset_identity"]+=1
        if row.get("split") not in ("train","development"):
            raise ValueError("only train/development source corrections may be used")
        declared["split_"+row["split"]]+=1
    if samples<1 or not declared["split_train"] or not declared["split_development"]:
        raise ValueError("no full original train+development correction bank")
    if strict and any(missing.values()):
        raise ValueError(
            "strict cell identity requires model_id, model_revision, and dataset/domain"
            " on EVERY correction row; missing "+str(dict(missing))
        )
    return {
        "schema":"eesd-iclr2027-model-dataset-data-identity-v1",
        "cell":cell,"model_id":profile["model_id"],
        "model_revision":profile["model_revision"],"dataset":profile["dataset"],
        "rows":samples,"source_sha256":hash_file(rows_path),
        "declared_fields":dict(declared),"missing_fields":dict(missing),
        "status":"verified_model_and_dataset_identity" if not missing else
                 "incomplete_record_identity_metadata",
        "training_support":"trainer source supports model-specific LoRA, raw GPU run not performed",
        "evaluation_support":profile["primary_evaluator"],
        "warning":"These checks cannot prove disjointness of training and evaluator problems. Run D01 with a real source-to-problem mapping."
    }


def coverage_rows(*,inputs_root:Path|None=None)->list[dict]:
    result=[]
    for cell,profile in load_matrix().items():
        f=(inputs_root/cell/"corrections.jsonl") if inputs_root else None
        if f is not None and f.is_file():
            try:
                audited=audit_correction_identity(cell,f)
                data_status=audited["status"]
                data_sha=audited["source_sha256"]
            except Exception as exc:
                data_status="INVALID: "+str(exc)
                data_sha=None
        else:
            data_status="not_supplied"
            data_sha=None
        adapter=(inputs_root/cell/"shared_initial_adapter"/"adapter_config.json") if inputs_root else None
        eval_ids=(inputs_root/cell/"eval_source_problem_ids.jsonl") if inputs_root else None
        evaluation=profile["primary_evaluator"]
        result.append({
            "cell":cell,"dataset":profile["dataset"],"model":profile["model"],
            "model_id":profile["model_id"],"revision":profile["model_revision"],
            "model_config":profile["model_config"],
            "training_code":"model-aware LoRA trainer implemented; H100 scientific run not verified",
            "correction_data":data_status,"correction_sha256":data_sha,
            "initial_adapter":"supplied_not_validated" if adapter and adapter.exists() else "not_supplied",
            "eval_id_map":"supplied_not_validated" if eval_ids and eval_ids.exists() else "not_supplied",
            "evaluation_code":evaluation,
            "evaluation_claim_allowed":False,
            "historical_no_update_solved":profile["original_paper"]["no_update_solved"],
            "historical_eed_solved":profile["original_paper"]["original_eed_solved"],
            "historical_N":profile["original_paper"]["N"],
        })
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs-root",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--cell",choices=sorted(load_matrix()))
    p.add_argument("--corrections",type=Path)
    p.add_argument("--strict-record-identity",action="store_true")
    args=p.parse_args()
    if args.cell:
        if args.corrections is None:
            p.error("--cell requires --corrections")
        data=audit_correction_identity(args.cell,args.corrections,strict=args.strict_record_identity)
    else:
        if args.corrections is not None:p.error("--corrections requires --cell")
        rows=coverage_rows(inputs_root=args.inputs_root)
        data={
            "schema":"eesd-iclr2027-coverage-inventory-v1",
            "original_model_count":len(MODELS),"original_dataset_count":len(DATASETS),
            "original_paper_cells":len(rows),
            "directly_implemented_heldout_evaluation_cells":sum(
                r["evaluation_code"]=="pinned_pbpf_sandbox" for r in rows),
            "heldout_protocol_unsupported_or_unqualified_cells":sum(
                r["evaluation_code"]!="pinned_pbpf_sandbox" for r in rows),
            "new_scientific_gpu_results":0,
            "cells":rows,
        }
    if args.output:
        if args.output.exists():raise FileExistsError("coverage report already exists")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
    print(json.dumps(data,indent=2,ensure_ascii=False))


if __name__=="__main__":main()
