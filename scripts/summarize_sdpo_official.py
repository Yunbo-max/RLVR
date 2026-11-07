#!/usr/bin/env python3
"""Summarize a complete upstream validation dump on the sealed Rich64 cohort."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from eesd.sdpo_official import summarize_validation

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--validation', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(summarize_validation(a.validation, a.data_dir), indent=2))
