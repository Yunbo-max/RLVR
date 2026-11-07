"""Self-contained Python entrypoint helper for individually runnable EED cases.

Can run each case directly with:
  python scripts/rebuttal_cases/c03_eed.py --cell codearc__deepseek_6p7b ...
This wrapper loads its own common audited implementation, then FIXES its
scientific case ID regardless of CLI. No batch other cases are launched.
"""
from __future__ import annotations
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=ROOT/"scripts"
SRC=ROOT/"src"
for p in (str(SRC),str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0,p)


def main_for(case_id:str):
    if not case_id or not (case_id.startswith("C") or case_id.startswith("D")):
        raise ValueError("registered independent case ID required")
    from run_eesd_rebuttal_case import main
    return main(selected_case=case_id)
