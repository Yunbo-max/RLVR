#!/usr/bin/env python3
"""Rebuttal-only, manuscript-faithful FOUR-transition Dirichlet-EESD weights.

DO NOT confuse with current src/eesd/distillation.py's THREE-state
axiomatic benefit method. ICLR 2027 submission #46512 uses:
  z = (FIX, REGRESS, PRESERVE, UNRESOLVED)
  v = (+1,-1,0,0), alpha=0.1, lambda=0.5
  w = [E[v^T theta] - lambda * SD(v^T theta)]+
  alpha_post = alpha + MASS * r; MASS=1/sum(p_i^2) or n.

This utility prepares *weight receipts only*; it does NOT train a model,
assert shared optimizer states, or produce downstream accuracy numbers.
Use separately verified, matched trainer/eval for rebuttal claims.

Input: original scored or unscored correction JSONL, required keys
split (train|development), source_component_id, trajectory_id,
before_outcomes, after_outcomes, relevance. Outcomes may be integers 0..4,
where 0 is PASS, or exact registry strings.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

from eesd.evidence import (correction_posterior, conservative_utility,
                           effective_mass, normalize_relevance)
from eesd.registry import OUTCOMES

UTIL=(1.0,-1.0,0.0,0.0)
WEIGHTS=("equal_weight","fixed_mass","adaptive_eed","eed_no_uncertainty",
         "constant_mass_matched","shuffled_eed","eed_terminal_quality",
         "final_correctness","scalar_confidence")
TEMPERATURE_SCALE=(0.5,1.0,2.0)


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as inp:
        for x in iter(lambda:inp.read(1048576),b""):h.update(x)
    return h.hexdigest()


def to_codes(row:dict,name:str)->list[int]:
    arr=row.get(name)
    if not isinstance(arr,list) or len(arr)<1:
        raise ValueError(f"missing nonempty {name}")
    out=[]
    for v in arr:
        if isinstance(v,str):
            if v not in OUTCOMES:raise ValueError(f"unknown execution outcome {v}")
            v=OUTCOMES.index(v)
        if type(v) is not int or not 0<=v<len(OUTCOMES):
            raise ValueError(f"bad execution code for {name}: {v!r}")
        out.append(v)
    return out


def pow_relevance(scores:list[float],scale:float)->list[float]:
    """Scale original exp(kappa*cos) strengths to another kappa, stably.

    Only legitimate IF raw relevance was s_i = exp(16 * cos_i), with all s_i>0.
    """
    if not math.isfinite(scale) or scale<=0:raise ValueError("invalid temperature scale")
    if len(scores)<1 or any(type(v) not in (int,float) or not math.isfinite(v)
        or v<=0 for v in scores):
        raise ValueError("temperature sweep requires positive kernel scores")
    logs=[scale*math.log(v) for v in scores]
    m=max(logs)
    return [math.exp(x-m) for x in logs]


def weights_for_record(before:list[int],after:list[int],relevance:list[float],*,
                       alpha:float=.1,lam:float=.5)->dict:
    if not isinstance(before,list) or not isinstance(after,list) or not isinstance(relevance,list):
        raise ValueError("invalid transition arrays")
    if len(before)<1 or len(before)!=len(after) or len(before)!=len(relevance):
        raise ValueError("mismatched transition/relevance arrays")
    fixed=correction_posterior(before,after,relevance,alpha=alpha,mass_rule="fixed")
    adaptive=correction_posterior(before,after,relevance,alpha=alpha,mass_rule="effective")
    fix=conservative_utility(fixed,UTIL,uncertainty_penalty=lam)
    eed=conservative_utility(adaptive,UTIL,uncertainty_penalty=lam)
    unpenalized=conservative_utility(adaptive,UTIL,uncertainty_penalty=0.)
    q=sum(v==0 for v in after)/len(after)
    return {
       "equal_weight":1.0,
       "fixed_mass":fix["positive_weight"],
       "adaptive_eed":eed["positive_weight"],
       "eed_no_uncertainty":unpenalized["positive_weight"],
       "eed_terminal_quality":eed["positive_weight"]*q,
       "final_correctness":float(q==1.0),
       "scalar_confidence":float(q),
       "absolute_final_public_pass_fraction":q,
       "effective_mass":effective_mass(relevance),
       "fixed_mass_value":len(relevance),
       "fixed_posterior":fixed.tolist(),
       "effective_posterior":adaptive.tolist(),
       "relative_relevance":normalize_relevance(relevance).tolist(),
       "posterior_mean":eed["mean"],
       "posterior_sd":eed["std"],
    }


def validate_raw(rows:list[dict])->list[dict]:
    if not rows:raise ValueError("no corrections supplied")
    ids=set()
    split_of_source={}
    normalized=[]
    for raw in rows:
        if not isinstance(raw,dict):raise ValueError("nonobject row")
        req={"split","source_component_id","trajectory_id",
             "before_outcomes","after_outcomes","relevance"}
        if req-set(raw):raise ValueError(f"missing fields: {req-set(raw)}")
        split=raw["split"]
        if split not in ("train","development"):
            raise ValueError("test sources must not enter rebuttal training-weight construction")
        source=raw["source_component_id"]
        tid=raw["trajectory_id"]
        if not isinstance(source,str) or not source or not isinstance(tid,str) or not tid:
            raise ValueError("invalid identity")
        if tid in ids:raise ValueError("repeated trajectory ID")
        ids.add(tid)
        if source in split_of_source and split_of_source[source]!=split:
            raise ValueError("train/dev overlapping source ID")
        split_of_source[source]=split
        before=to_codes(raw,"before_outcomes");after=to_codes(raw,"after_outcomes")
        relevance=raw["relevance"]
        if (len(before)!=len(after) or not isinstance(relevance,list)
            or len(relevance)!=len(after)
            or any(type(v) not in (int,float) or not math.isfinite(v) or v<0
                   for v in relevance) or sum(relevance)<=0):
            raise ValueError("invalid relevance/outcome sizes or weights")
        normalized.append({**raw,"before_outcomes":before,"after_outcomes":after,
                           "relevance":relevance})
    if not any(x["split"]=="train" for x in normalized):
        raise ValueError("missing train examples")
    return normalized


def add_weights(rows:list[dict],*,alpha:float=.1,lam:float=.5,
                seed:int=1701,include_kappa_sweep:bool=True,
                kappa:float=16.)->tuple[list[dict],dict]:
    if not math.isfinite(kappa) or kappa<=0:
        raise ValueError("kappa must be finite and positive")
    normalized=validate_raw(rows)
    annotated=[]
    for row in normalized:
        # Power-transform only the ORIGINAL s_i=exp(16*cos) kernel scores.
        # Scaling `log(s_i)` by kappa/16 retains the exact relevance model;
        # it is NOT an ablation of the temperature without source provenance.
        relevance=(row["relevance"] if kappa==16. else
                   pow_relevance(row["relevance"],kappa/16.))
        exact=weights_for_record(row["before_outcomes"],row["after_outcomes"],
            relevance,alpha=alpha,lam=lam)
        w={k:float(exact[k]) for k in
           ("equal_weight","fixed_mass","adaptive_eed","eed_no_uncertainty",
            "eed_terminal_quality","final_correctness","scalar_confidence")}
        detail={k:v for k,v in exact.items() if k not in w}
        detail["kappa_used_for_current_weights"]=kappa
        detail["source_relevance_kernel_kappa"]=16.0
        if include_kappa_sweep:
            detail["kappa_relative_sweep"]={}
            for factor in TEMPERATURE_SCALE:
                varied=weights_for_record(row["before_outcomes"],row["after_outcomes"],
                    pow_relevance(row["relevance"],factor),alpha=alpha,lam=lam)
                detail["kappa_relative_sweep"][str(16*factor)]={
                    "effective_mass":varied["effective_mass"],
                    "adaptive_weight":varied["adaptive_eed"],
                    "zero_weight":varied["adaptive_eed"]==0.0}
        annotated.append({**row,"rebuttal_weights":w,
                          "rebuttal_diagnostics":detail})
    # Dev is NOT used to select a constant training-strength coefficient.
    train=[r["rebuttal_weights"]["adaptive_eed"] for r in annotated if r["split"]=="train"]
    mean=sum(train)/len(train)
    if not 0<=mean<=1:raise ValueError("invalid constant weight")
    for r in annotated:r["rebuttal_weights"]["constant_mass_matched"]=mean
    # Permute within split (training distribution preserved), never use
    # evaluation results to choose weights. Deterministic, nontrivial rotation.
    for split in ("train","development"):
        selected=sorted((r for r in annotated if r["split"]==split),
                        key=lambda r:r["trajectory_id"])
        original=[r["rebuttal_weights"]["adaptive_eed"] for r in selected]
        if len(original)>1:
            rng=random.Random(f"{seed}:{split}:shuffled_EED")
            shift=rng.randint(1,len(original)-1)
            reordered=original[shift:]+original[:shift]
        else:reordered=original
        for r,v in zip(selected,reordered):
            r["rebuttal_weights"]["shuffled_eed"]=v
    summary={
        "schema":"eesd-submitted-four-state-rebuttal-weights-v1",
        "method":"ICLR2027 original Dirichlet EESD, not new 3-state axiomatic EESD",
        "utility":list(UTIL),"alpha":alpha,"uncertainty_penalty":lam,
        "kappa":kappa,"source_kernel_kappa":16.,
        "seed":seed,"row_counts":dict(Counter(r["split"] for r in annotated)),
        "source_counts":{split:len({r["source_component_id"] for r in annotated
                                  if r["split"]==split}) for split in ("train","development")},
        "mean_matched_weight_from_train_only":mean,
        "arms":list(WEIGHTS),
        "training_note":"weights_only; dedicated exact-matched KL/CE training is REQUIRED",
        "zero_weight_note":"zeros must retain KL on their original examples, as in submission",
        "temperature_sweep_note":"kappa 8/16/32 valid only if relevance scores use exp(16*cos)",
    }
    summary["train_stats"]={
        arm:{"mean":sum(r["rebuttal_weights"][arm]
                         for r in annotated if r["split"]=="train")/len(train),
             "zero":sum(r["rebuttal_weights"][arm]==0
                        for r in annotated if r["split"]=="train")}
        for arm in WEIGHTS}
    return annotated,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--alpha",type=float,default=.1)
    p.add_argument("--lambda-penalty",type=float,default=.5)
    p.add_argument("--seed",type=int,default=1701)
    p.add_argument("--kappa",type=float,choices=(8.,16.,32.),default=16.)
    p.add_argument("--skip-kappa-sweep",action="store_true")
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("output already exists")
    original=[json.loads(s) for s in a.input.read_text().split("\n") if s.strip()]
    scored,manifest=add_weights(original,alpha=a.alpha,lam=a.lambda_penalty,
                                seed=a.seed,include_kappa_sweep=not a.skip_kappa_sweep,
                                kappa=a.kappa)
    a.output.mkdir(parents=True,exist_ok=False)
    with (a.output/"weights.jsonl").open("x") as file:
        for row in scored:file.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
    manifest["source_file_sha256"]=sha(a.input)
    manifest["weights_file_sha256"]=sha(a.output/"weights.jsonl")
    (a.output/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"rows":len(scored),"arms":manifest["train_stats"]},indent=2))


if __name__=="__main__":main()
