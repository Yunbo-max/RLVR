#!/usr/bin/env python3
"""Prepare or run one prospective matched EESD rebuttal evaluation.

This common greedy NF4/BF16 sandbox protocol is separate from historical PBPF
replication. Preparation needs frozen local inputs; execution uses the paths
sealed in that case's evaluation/plan.json. No completed output is overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eesd import rebuttal_eval as ev


def check_run_options(args, plan):
    """Optional run arguments may restate a frozen input, never replace it."""
    fields = {
        "pbpf_root": "pbpf_root", "public_root": "public_root",
        "evaluator_root": "evaluator_root", "eval_ids": "eval_source_id_file_path",
    }
    for argument, field in fields.items():
        supplied = getattr(args, argument)
        if supplied is not None and supplied.resolve() != Path(plan[field]).resolve():
            raise ValueError(f"--{argument.replace('_', '-')} differs from the frozen evaluation plan")
    for field in ("domain", "workers"):
        supplied = getattr(args, field)
        if supplied is not None and supplied != plan[field]:
            raise ValueError(f"--{field} differs from the frozen evaluation plan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--pbpf-root", type=Path)
    parser.add_argument("--public-root", type=Path)
    parser.add_argument("--evaluator-root", type=Path)
    parser.add_argument("--eval-ids", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--domain", choices=sorted(set(ev.DATASETS.values())))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--action", choices=("prepare", "run", "both"), default=None)
    action.add_argument("--_phase", choices=("generate", "evaluate"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    case = args.case_dir.resolve()
    operation = args.action or "prepare"

    if args._phase:
        plan = ev.read_json(case / "evaluation/plan.json")
        check_run_options(args, plan)
        result = getattr(ev, args._phase)(case)
    else:
        if operation in ("prepare", "both"):
            required = ("pbpf_root", "public_root", "evaluator_root", "eval_ids")
            missing = ["--" + name.replace("_", "-") for name in required if getattr(args, name) is None]
            if missing:
                parser.error("preparation requires " + ", ".join(missing))
            result = ev.prepare(
                case=case, pbpf=args.pbpf_root, public=args.public_root,
                evaluator=args.evaluator_root, eval_ids=args.eval_ids,
                workers=1 if args.workers is None else args.workers, domain=args.domain,
            )
        if operation in ("run", "both"):
            plan = ev.read_json(case / "evaluation/plan.json")
            check_run_options(args, plan)
            result = ev.run(case)

    summary = {"case_dir": str(case), "action": args._phase or operation}
    if isinstance(result, dict):
        for key in ("schema", "status", "evaluation_protocol", "case_id", "evaluation_sources", "sources"):
            if key in result:
                summary[key] = result[key]
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
