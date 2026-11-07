#!/usr/bin/env python3
"""Export public-only Rich64 training and disjoint final evaluation for upstream SDPO."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from eesd.sdpo_official import export_cohort

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-limit', type=int, default=64)
    p.add_argument('--eval-limit', type=int, default=64)
    a = p.parse_args()
    print(json.dumps(export_cohort(a.data_dir, a.output, train_limit=a.train_limit,
                                  eval_limit=a.eval_limit), indent=2))
