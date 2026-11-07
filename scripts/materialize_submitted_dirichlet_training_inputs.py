#!/usr/bin/env python3
"""Materialize identical correction inputs for manuscript-faithful EED ablations.

Each output arm stores the SAME raw corrections and order and changes ONLY
the per-trajectory supervised CE weight. We place the chosen weight into
the anchored "eesd_full" execution slot of the existing trainer; the true
scientific rule name is carried separately in the manifest and each row.

Run each input with:
  python scripts/run_eesd_weighted_sft.py --input <arm>/scored-corrections.jsonl
    --rule eesd_full --include-zero-weight-kl --anchor-beta 0.03
    --model-config <pinned_model.yaml> --seed <same_seed>
    --response-token-budget <same_budget> --output <new_arm_output>

IMPORTANT: the trainer's rule NAME is only an execution slot; each
manifest states whether the data contains equal/fixed/adaptive/shuffled
original four-state weights. This does NOT reproduce old weights by
calling the current three-state score_trajectory routine.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from eesd.distillation import TRAIN_RULES,validate_scored_record

ARMS=("equal_weight","fixed_mass","adaptive_eed","eed_no_uncertainty",
      "constant_mass_matched","shuffled_eed","eed_terminal_quality")


def hash_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()


def materialize(source:Path, output:Path,*,arms:tuple[str,...]=ARMS)->dict:
    if output.exists():raise FileExistsError("refuse to overwrite rebuttal training inputs")
    rows=[json.loads(line) for line in source.read_text().split("\n") if line.strip()]
    if not rows:raise ValueError("empty rebuttal weight file")
    identities=[(x["trajectory_id"],x["source_component_id"],x["split"]) for x in rows]
    if len({a for a,_,_ in identities})!=len(identities):
        raise ValueError("repeated trajectory")
    if not any(x["split"]=="train" for x in rows) or not any(
        x["split"]=="development" for x in rows):
        raise ValueError("nonempty train + dev rows needed")
    for row in rows:
        for key in ("prompt","correction","before_outcomes","after_outcomes",
                    "relevance","rebuttal_weights"):
            if key not in row:raise ValueError(f"missing correction detail: {key}")
        if any(arm not in row["rebuttal_weights"] for arm in arms):
            raise ValueError("missing selected manuscript weights")
    outputs={}
    output.mkdir(parents=True,exist_ok=False)
    for arm in arms:
        folder=output/arm;folder.mkdir(exist_ok=False)
        result=folder/"scored-corrections.jsonl"
        with result.open("x") as file:
            for original in rows:
                weight=float(original["rebuttal_weights"][arm])
                if not 0<=weight<=1:raise ValueError("invalid rebuttal weight")
                # Slot maps to the ANCHORED training implementation. Other
                # registry keys must exist for validation but are UNUSED.
                slots={rule:0.0 for rule in TRAIN_RULES}
                slots["eesd_full"]=weight
                record={**original,
                    "training_weights":slots,
                    "manuscript_rebuttal_arm":arm,
                    "training_slot":"eesd_full"}
                validate_scored_record(record)
                file.write(json.dumps(record,sort_keys=True,ensure_ascii=False)+"\n")
        outputs[arm]={"path":str(result),"sha256":hash_file(result),
                      "train_positive":sum(x["split"]=="train" and
                          x["rebuttal_weights"][arm]>0 for x in rows),
                      "train_zero":sum(x["split"]=="train" and
                          x["rebuttal_weights"][arm]==0 for x in rows)}
    return {"schema":"submitted-dirichlet-matched-train-inputs-v1",
        "source_weights_sha256":hash_file(source),
        "raw_row_count":len(rows),
        "identity_order_sha256":hashlib.sha256(
            json.dumps(identities,ensure_ascii=False).encode()).hexdigest(),
        "arms":outputs,
        "trainer":"scripts/run_eesd_weighted_sft.py",
        "trainer_rule_slot":"eesd_full",
        "training_requires":"--include-zero-weight-kl --anchor-beta 0.03",
        "provenance":"original FOUR-state submitted EED, NOT current 3-state scorer",
        "scientific_result":"none; these files only prepare identical train inputs"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS))
    a=p.parse_args()
    if len(set(a.arms))!=len(a.arms):raise ValueError("duplicate arms")
    report=materialize(a.weights,a.output,arms=tuple(a.arms))
    (a.output/"manifest.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({x:y["train_positive"] for x,y in report["arms"].items()},indent=2))


if __name__=="__main__":main()
