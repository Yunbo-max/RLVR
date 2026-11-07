#!/usr/bin/env python3
"""Reviewer-requested audit of RAW future-outcome labels and matched NLL.

Fail-closed required per-row schema:
  source_component_id, problem_id, seed, visible_test_ids (list of IDs),
  future_test_id, future_outcome (one string), outcome_classes (same ordered
  >=2 distinct class strings for both distributions), fixed_probs,
  adaptive_probs (same ordered normalized categorical probabilities),
  n_visible, predictor_lock_sha256 (same model/hyperparameters OTHER THAN mass).

Per-source visible/future test IDs must be disjoint. Reports raw aggregate
scores BY seed and n_visible; matching labels and model configuration
cannot be inferred from the submission's aggregate CSV and must be
recovered from genuine earlier experiment receipts.

The script never executes code or updates policies. Source-cluster bootstrap
accounts for shared benchmark tasks; DO NOT substitute training labels for
unseen future outcomes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()


def _probabilities(value,classes,name):
    if not isinstance(value,list) or len(value)!=len(classes):
        raise ValueError(f"{name} must have one probability per outcome class")
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 or v>1
           for v in value):
        raise ValueError(f"{name} must be nonnegative finite probabilities")
    if not math.isclose(sum(value),1.0,abs_tol=1e-6,rel_tol=0):
        raise ValueError(f"{name} probabilities must sum to one")
    return np.asarray(value,dtype=float)


def parse_rows(path:Path)->list[dict]:
    rows=[json.loads(x) for x in path.read_text().split("\n") if x.strip()]
    if not rows:raise ValueError("no raw future-outcome rows")
    classes=None
    identities=set()
    lock_per_seed={}
    by_source=defaultdict(lambda:{"visible":set(),"future":set()})
    normalized=[]
    for r in rows:
        required=("source_component_id","problem_id","seed",
                  "visible_test_ids","future_test_id","future_outcome",
                  "outcome_classes","fixed_probs","adaptive_probs",
                  "n_visible","predictor_lock_sha256")
        missing=[x for x in required if x not in r]
        if missing:raise ValueError(f"missing raw future-outcome fields: {missing}")
        sid,pid=r["source_component_id"],r["problem_id"]
        if not isinstance(sid,str) or not sid or pid is None or str(pid)=="":
            raise ValueError("source and problem identities must be supplied")
        seed=r["seed"]
        if type(seed)is not int:raise ValueError("integer initialization seed required")
        key=(seed,sid,r["future_test_id"])
        if key in identities:raise ValueError("duplicate source/seed/future test")
        identities.add(key)
        n=r["n_visible"]
        visible=r["visible_test_ids"]
        if type(n)is not int or n<1 or not isinstance(visible,list) or len(visible)!=n:
            raise ValueError("n_visible must match number of visible test IDs")
        if any(not isinstance(t,(str,int)) for t in visible):
            raise ValueError("visible test IDs must be string/integer")
        visible_ids=[str(t) for t in visible]
        future=str(r["future_test_id"])
        if len(set(visible_ids))!=len(visible_ids) or future in visible_ids:
            raise ValueError("future outcome test leaks into visible tests")
        row_classes=r["outcome_classes"]
        if (not isinstance(row_classes,list) or len(row_classes)<2
            or any(not isinstance(x,str) or not x for x in row_classes)
            or len(row_classes)!=len(set(row_classes))):
            raise ValueError("fixed categorical outcome vocabulary required")
        if classes is not None and classes!=row_classes:
            raise ValueError("prediction outcome category ordering changed")
        classes=row_classes
        label=r["future_outcome"]
        if label not in classes:
            raise ValueError("unrecognized future-outcome label")
        p_fixed=_probabilities(r["fixed_probs"],classes,"fixed_probs")
        p_adapt=_probabilities(r["adaptive_probs"],classes,"adaptive_probs")
        lock=r["predictor_lock_sha256"]
        if not isinstance(lock,str) or len(lock)!=64 or any(ch not in "0123456789abcdef" for ch in lock):
            raise ValueError("reproducible predictor configuration lock required")
        if seed in lock_per_seed and lock_per_seed[seed]!=lock:
            raise ValueError("predictor hyperparameters not matched within seed")
        lock_per_seed[seed]=lock
        source_key=(seed,sid)
        # Test bank IDs can appear as visible in some source rows and future in
        # others. For a strict future generalization claim, veto such reuse.
        by_source[source_key]["visible"].update(visible_ids)
        by_source[source_key]["future"].add(future)
        e_mass=r.get("effective_mass",None)
        if e_mass is not None and (
            type(e_mass)not in (int,float) or not math.isfinite(e_mass)
            or e_mass<1 or e_mass>n+1e-8):
            raise ValueError("effective mass outside [1,n_visible]")
        idx=classes.index(label)
        normalized.append({
            "seed":seed,"source_component_id":sid,"problem_id":str(pid),
            "n_visible":n,"future_test_id":future,
            "label":label,"label_index":idx,
            "fixed_probs":p_fixed,"adaptive_probs":p_adapt,
            "effective_mass":float(e_mass) if e_mass is not None else None,
            "predictor_lock_sha256":lock,
        })
    for key,bank in by_source.items():
        if bank["visible"]&bank["future"]:
            raise ValueError(f"future test reused as visible for {key}")
    return normalized


def _summarize_subset(rows:list[dict],*,bootstrap=2000,seed=314159)->dict:
    if not rows:raise ValueError("empty prediction subset")
    y=np.asarray([r["label_index"] for r in rows],dtype=int)
    fixed=np.stack([r["fixed_probs"] for r in rows])
    adapt=np.stack([r["adaptive_probs"] for r in rows])
    n=len(y)
    eps=1e-12
    fixed_nll=-np.log(np.maximum(fixed[np.arange(n),y],eps))
    adapt_nll=-np.log(np.maximum(adapt[np.arange(n),y],eps))
    fixed_acc=(fixed.argmax(axis=1)==y)
    adapt_acc=(adapt.argmax(axis=1)==y)
    truth=np.eye(fixed.shape[1])[y]
    diff=fixed_nll-adapt_nll
    problem_groups=defaultdict(list)
    for i,r in enumerate(rows):problem_groups[r["problem_id"]].append(i)
    problems=sorted(problem_groups)
    problem_diffs=np.asarray([diff[problem_groups[k]].mean() for k in problems])
    rng=np.random.default_rng(seed)
    sample=rng.integers(0,len(problems),size=(bootstrap,len(problems)))
    ci=np.quantile(problem_diffs[sample].mean(axis=1),[.025,.975])
    return {
        "observations":n,"independent_problem_ids":len(problems),
        "fixed_NLL":float(fixed_nll.mean()),"adaptive_NLL":float(adapt_nll.mean()),
        "fixed_minus_adaptive_NLL":float(diff.mean()),
        "problem_mean_NLL_gain":float(problem_diffs.mean()),
        "problem_cluster_gain_ci95":[float(ci[0]),float(ci[1])],
        "fixed_accuracy":float(fixed_acc.mean()),"adaptive_accuracy":float(adapt_acc.mean()),
        "argmax_agreement":float((fixed.argmax(axis=1)==adapt.argmax(axis=1)).mean()),
        "fixed_Brier":float(((fixed-truth)**2).sum(axis=1).mean()),
        "adaptive_Brier":float(((adapt-truth)**2).sum(axis=1).mean()),
    }


def summarize(path:Path,*,bootstrap:int=2000,seed:int=314159)->dict:
    if bootstrap<100:raise ValueError("at least 100 bootstrap draws")
    rows=parse_rows(path)
    seeds=sorted(set(r["seed"] for r in rows))
    by_seed={}
    for value in seeds:
        sub=[r for r in rows if r["seed"]==value]
        nvals=sorted(set(r["n_visible"] for r in sub))
        by_seed[str(value)]={
            "predictor_lock_sha256":sub[0]["predictor_lock_sha256"],
            "all":_summarize_subset(sub,bootstrap=bootstrap,seed=seed),
            "by_visible_count":{str(n):_summarize_subset(
                [r for r in sub if r["n_visible"]==n],bootstrap=bootstrap,seed=seed)
                 for n in nvals},
            "mass_concentration_bins":{},
        }
        # Emphasize concentration-stratified NLL only if raw E(p) is supplied.
        mass_rows=[r for r in sub if r["effective_mass"] is not None]
        if len(mass_rows)==len(sub):
            for start,stop in ((1,2),(2,3),(3,4),(4,math.inf)):
                interval=[r for r in sub
                          if start<=r["effective_mass"]<stop or (
                              stop==math.inf and r["effective_mass"]>=start)]
                if interval:
                    by_seed[str(value)]["mass_concentration_bins"][f"[{start},{stop})"]=_summarize_subset(
                        interval,bootstrap=bootstrap,seed=seed)
    # Detect same task population across predictor initializations, not assumed.
    populations={k:{(r["source_component_id"],r["future_test_id"])
                    for r in rows if str(r["seed"])==k} for k in by_seed}
    keys=list(populations)
    matched=all(populations[k]==populations[keys[0]] for k in keys[1:])
    return {
        "schema":"eesd-four-state-raw-future-outcome-audit-v1",
        "status":"computed_from_raw_future_labels",
        "input_sha256":digest(path),"outcome_classes":rows[0]["fixed_probs"].size,
        "outcome_class_names":next(json.loads(s)["outcome_classes"] for s in path.read_text().split("\n") if s.strip()),
        "seeds":seeds,"predictor_initializations":len(seeds),
        "identical_future_test_populations_across_seeds":matched,
        "results":by_seed,
        "critical_limitations":[
            "Valid raw observation IDs and predictions do not alone establish train/eval source disjointness.",
            "Label semantics still require a human documented definition and dataset release provenance.",
            "NLL of an outcome predictor does not prove improved policy updates.",
            "Source bootstrap is conditional on predictor models, not variance over training seeds."
        ]
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=2000)
    p.add_argument("--seed",type=int,default=314159)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("raw predictor audit already exists")
    report=summarize(a.input,bootstrap=a.bootstrap,seed=a.seed)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":report["status"],
                      "seeds":report["seeds"],
                      "population_matched":report["identical_future_test_populations_across_seeds"]},indent=2))


if __name__=="__main__":main()
