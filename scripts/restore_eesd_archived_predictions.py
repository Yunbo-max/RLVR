#!/usr/bin/env python3
"""Restore D03 observations from the pinned PBPF cache and prediction archives.

Recomputes the two matched predictor arms from the recorded public history and
future input queries, then requires agreement with the archived NPZ. Does not
execute programs, regenerate outcomes or fit a model. This verifies the saved
cache/NPZ relationship, not the unavailable original execution banks themselves.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import audit_eesd_future_outcome_pairs as audit

PBPF_PIN="0200283849274b49e303638b91f142acd8bba984"
RUNNER_SHA="09d016fb8b9caf6c8b7829a56550ce99f4d1fbefb83a1091a60396a6026aff85"
EVIDENCE_SHA="f825e0a70a41d2c9e3a8b96aafc89ea54d70c6b36337b90697c906e93758d7be"
ARCHIVE="artifacts/eesd-eval-20260921/mechanism"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(ok,message):
    if not ok:raise ValueError(message)


def observation_rows(records,*,seed,classes,labels,clusters,fixed,adaptive,masses,lock,visible):
    """Join archived arrays to original ordered cache IDs without inventing IDs."""
    require(len(labels)==len(clusters)==len(fixed)==len(adaptive)==len(masses),
            "saved array lengths differ")
    result=[];i=0;sources=set()
    for row in records:
        sid=row.get("source_component_id");pid=row.get("problem_id")
        require(isinstance(sid,str) and sid and isinstance(pid,str) and pid,"missing original source/problem identity")
        require(sid not in sources,"duplicate cache source")
        sources.add(sid)
        tests=row["tests"];outcomes=row["outcomes"]
        ids=[str(t["id"]) for t in tests]
        require(len(tests)==len(outcomes) and len(tests)>visible and len(set(ids))==len(ids),
                "invalid or overlapping original visible/future test IDs")
        for j in range(visible,len(tests)):
            require(i<len(labels),"cache contains more future observations than NPZ")
            require(str(clusters[i])==sid and classes[int(labels[i])]==outcomes[j],
                    "saved labels/source order does not match original cache")
            result.append({
                "source_component_id":sid,"problem_id":pid,"seed":seed,
                "visible_test_ids":ids[:visible],"future_test_id":ids[j],
                "future_outcome":outcomes[j],"outcome_classes":list(classes),
                "fixed_probs":fixed[i].tolist(),"adaptive_probs":adaptive[i].tolist(),
                "n_visible":visible,"effective_mass":float(masses[i]),
                "predictor_lock_sha256":lock,"original_candidate_id":row["task_id"],
            })
            i+=1
    require(i==len(labels),"NPZ contains extra observations absent from cache")
    return result


def load_runner(pbpf:Path):
    head=subprocess.run(["git","rev-parse","HEAD"],cwd=pbpf,text=True,capture_output=True,check=True).stdout.strip()
    require(head==PBPF_PIN,"wrong pinned PBPF revision")
    dirty=subprocess.run(["git","status","--porcelain"],cwd=pbpf,text=True,capture_output=True,check=True).stdout.strip()
    require(not dirty,"PBPF source checkout must remain clean")
    runner=pbpf/"scripts/run_eesd_evidence_matrix.py"
    require(sha(runner)==RUNNER_SHA and sha(pbpf/"src/pbpf/eesd/evidence.py")==EVIDENCE_SHA,
            "frozen evidence source hash mismatch")
    sys.path.insert(0,str(pbpf/"src"))
    loaded=runpy.run_path(str(runner))
    from pbpf.eesd import evidence as imported_evidence
    require(sha(imported_evidence.__file__)==EVIDENCE_SHA,"imported PBPF evidence source mismatch")
    return loaded


def restore_seed(pbpf:Path,runner,*,dataset:str,family:str,seed:int):
    base=pbpf/ARCHIVE/dataset/family/f"seed{seed}"
    report_path=base/"report/report.json";npz_path=base/"report/predictions.npz"
    cache_path=base/"cache.json";seal_path=base/"report/complete.json"
    report=read(report_path);cache=read(cache_path);seal=read(seal_path)
    config_path=pbpf/"configs/experiments/eesd_iclr2027.yaml"
    require(sha(config_path)==report.get("config_sha256"),"archived experiment config checksum mismatch")
    config=yaml.safe_load(config_path.read_text())
    require(sha(report_path)==seal.get("report_sha256") and sha(npz_path)==seal.get("predictions_sha256"),
            "archived report/NPZ completion checksum mismatch")
    require(sha(cache_path)==report.get("cache_sha256"),"archived cache checksum mismatch")
    require(report.get("schema")=="eesd-evidence-matrix-v1" and report.get("dataset")==dataset
            and report.get("model")==family and report.get("seed")==seed and report.get("visible")==4
            and report.get("assessment_split")=="primary" and report.get("validation_split")=="development",
            "archived report identity/protocol mismatch")
    require(report.get("runner_source_sha256")==RUNNER_SHA and report.get("evidence_source_sha256")==EVIDENCE_SHA,
            "archived report uses another predictor implementation")
    identity=cache.get("generator_identity")
    require(isinstance(identity,list) and len(identity)==4 and identity[2] is None and identity[3]==seed,
            "archived cache generator identity mismatch")
    model_role={"qwen25_7b":"primary","deepseek_6p7b":"cross_family"}[family]
    model_config=yaml.safe_load((pbpf/config["models"][model_role]).read_text())
    require(identity[:2]==[model_config["model_id"],model_config["revision"]],
            "archived cache generator model/revision mismatch")
    require(cache.get("schema")=="eesd-public-query-mechanism-cache-v1" and cache.get("tests_per_candidate")==10,
            "unexpected historical cache schema")
    primary=[r for r in cache["records"] if r["split"]=="primary"]
    development=[r for r in cache["records"] if r["split"]=="development"]
    require(len(primary)==500 and len(development)==200,"original 500/200 cache population mismatch")
    for field in ("source_component_id","problem_id"):
        require(not ({r[field] for r in primary}&{r[field] for r in development}),
                "archived development/primary identities overlap")
    selected=report["selected"]["effective"]
    memo={}
    def development_examples(split,visible,strength,binary):
        require(split=="validation" and visible==4 and binary is False,
                "unexpected development selection protocol")
        if strength not in memo:
            memo[strength]=runner["build_examples"](development,visible,strength,binary=False)
        return memo[strength]
    evidence=config["evidence"]
    recomputed,grid=runner["select_arm"](
        development_examples,"validation",4,evidence["alphas"],evidence["strengths"],
        "effective",ece_bins=evidence["ece_bins"])
    saved_grid=report["selection_grids"]["effective"]
    require(len(grid)==len(saved_grid),"development selection grid size mismatch")
    for actual,saved in [(recomputed,selected),*zip(grid,saved_grid)]:
        require(set(actual)==set(saved),"development selection schema mismatch")
        for key,value in actual.items():
            require((value is None and saved[key] is None) or
                    (value is not None and saved[key] is not None and
                     np.isclose(value,saved[key],rtol=1e-12,atol=1e-14)),
                    f"development selection {key} mismatch")
    # A matched mass contrast shares alpha and lexical strength exactly.
    examples=runner["build_examples"](primary,4,selected["strength"],binary=False)
    fixed,_=runner["predictions"](examples,selected["alpha"],"fixed")
    effective,masses=runner["predictions"](examples,selected["alpha"],"effective")
    labels,clusters=runner["labels_clusters"](examples)
    with np.load(npz_path,allow_pickle=False) as saved:
        require(np.array_equal(labels,saved["labels"]) and np.array_equal(clusters,saved["clusters"]),
                "recomputed labels/source order differs from saved NPZ")
        for name,value in (("fixed_at_effective_params",fixed),("effective",effective),("effective_mass",masses)):
            require(value.shape==saved[name].shape and np.all(np.isfinite(saved[name]))
                    and np.allclose(value,saved[name],rtol=1e-12,atol=1e-14),
                    f"reconstructed {name} does not match saved probabilities")
        # Retain exact archived values after the reconstruction check.
        fixed=saved["fixed_at_effective_params"].copy();effective=saved["effective"].copy()
        masses=saved["effective_mass"].copy()
    lock_data={"generator_identity":identity,"alpha":selected["alpha"],
               "lexical_strength":selected["strength"],"visible":4,
               "outcome_classes":list(runner["OUTCOMES"]),
               "runner_sha256":RUNNER_SHA,"evidence_sha256":EVIDENCE_SHA}
    lock=hashlib.sha256(json.dumps(lock_data,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    rows=observation_rows(primary,seed=seed,classes=runner["OUTCOMES"],labels=labels,
                          clusters=clusters,fixed=fixed,adaptive=effective,masses=masses,lock=lock,visible=4)
    binding=read(base/"direct-binding.json")
    require(binding.get("cache_sha256")==sha(cache_path),"historical execution binding differs from cache")
    receipt={
        "dataset":dataset,"model":family,"seed":seed,"observations":len(rows),
        "source_directory":str(base.relative_to(pbpf)),
        "report_sha256":sha(report_path),"predictions_sha256":sha(npz_path),
        "cache_sha256":sha(cache_path),"completion_sha256":sha(seal_path),
        "experiment_config_sha256":sha(config_path),
        "effective_arm_development_grid_revalidated":True,
        "effective_arm_development_selection":selected,
        "predictor_lock":lock_data,"predictor_lock_sha256":lock,
        "historical_execution_profile":binding.get("execution_profile"),
        "reconstruction":"matched fixed/effective probabilities, labels, masses and cache ordering verified",
        "original_execution_bank_revalidated":False,
    }
    return rows,receipt


def recover(pbpf:Path,output:Path,*,bootstrap:int=10000):
    if output.exists():raise FileExistsError("archive recovery output already exists")
    if bootstrap<100:raise ValueError("at least 100 bootstrap draws")
    runner=load_runner(pbpf)
    output.mkdir(parents=True,exist_ok=False)
    cells={}
    for dataset in ("runbugrun","codearc"):
        for family in ("qwen25_7b","deepseek_6p7b"):
            cell=f"{dataset}__{family}";rows=[];receipts=[]
            for seed in (1701,1702,1703):
                seed_rows,receipt=restore_seed(pbpf,runner,dataset=dataset,family=family,seed=seed)
                rows.extend(seed_rows);receipts.append(receipt)
                print(json.dumps({"restored":cell,"seed":seed,"observations":len(seed_rows)}),flush=True)
            path=output/(cell+".jsonl")
            with path.open("x") as stream:
                for row in rows:stream.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
            summary=audit.summarize(path,bootstrap=bootstrap)
            cells[cell]={"raw_rows_file":path.name,"raw_rows_sha256":sha(path),
                         "restoration_receipts":receipts,"audit":summary}
    summary={
        "schema":"eesd-archived-predictor-recovery-v1",
        "status":"restored_and_recomputed_from_pinned_archived_cache",
        "pbpf_commit":PBPF_PIN,"cells":cells,"raw_observations":sum(
            r["observations"] for c in cells.values() for r in c["restoration_receipts"]),
        "bootstrap_draws":bootstrap,"bootstrap_seed":314159,
        "restoration_source_sha256":sha(__file__),
        "audit_source_sha256":sha(audit.__file__),
        "analysis_versions":{"numpy":np.__version__,"PyYAML":yaml.__version__},
        "seed_role":"frozen candidate-generation seeds; predictor parameters selected on each development cache; not downstream LoRA fitting seeds",
        "limitations":[
            "This is reanalysis of historical predictor outputs, not new GPU training or downstream policy evaluation.",
            "The cache records source/problem/test IDs; original external execution banks and dataset materializations have not been independently revalidated.",
            "Historical candidate generation saw four public tests; the probability predictor also received future test INPUT queries, never future outcomes as features.",
            "The predictor has five execution-outcome classes; it is distinct from the four correction-transition states used to weight correction training.",
            "The three archived seeds identify candidate banks, not three new downstream optimizer runs.",
            "History-size sweeps change both visible history and the future target set; their full NLL reduction cannot be attributed only to evidence mass.",
        ],
    }
    with (output/"summary.json").open("x") as stream:
        json.dump(summary,stream,indent=2,sort_keys=True);stream.write("\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pbpf-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=10000)
    args=p.parse_args()
    result=recover(args.pbpf_root.resolve(),args.output.resolve(),bootstrap=args.bootstrap)
    print(json.dumps({"status":result["status"],"raw_observations":result["raw_observations"]}))


if __name__=="__main__":main()
