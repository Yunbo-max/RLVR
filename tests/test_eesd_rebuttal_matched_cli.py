"""The CLI cannot redirect a sealed evaluation to other data or runtime inputs."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_eesd_rebuttal_matched_evaluation.py"
spec = importlib.util.spec_from_file_location("matched_cli", SCRIPT)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def prepared(tmp_path):
    case = tmp_path / "case"
    (case / "evaluation").mkdir(parents=True)
    plan = {
        "pbpf_root": str(tmp_path / "pbpf"), "public_root": str(tmp_path / "public"),
        "evaluator_root": str(tmp_path / "private"),
        "eval_source_id_file_path": str(tmp_path / "ids.jsonl"),
        "workers": 1, "domain": "rbr",
    }
    (case / "evaluation/plan.json").write_text(json.dumps(plan))
    return case, plan


@pytest.mark.parametrize("flag,value", [
    ("--pbpf-root", "/other/source"), ("--public-root", "/other/public"),
    ("--evaluator-root", "/other/private"), ("--eval-ids", "/other/ids"),
    ("--domain", "codearc"), ("--workers", "2"),
])
def test_run_rejects_overriding_frozen_inputs_before_execution(tmp_path, monkeypatch, flag, value):
    case, _ = prepared(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--case-dir", str(case), "--action", "run", flag, value])
    monkeypatch.setattr(cli.ev, "run", lambda *a: pytest.fail("execution must not start"), raising=False)
    with pytest.raises(ValueError, match="differs from the frozen"):
        cli.main()


def test_run_can_restate_identical_frozen_options(tmp_path, monkeypatch, capsys):
    case, plan = prepared(tmp_path)
    calls = []
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--case-dir", str(case), "--action", "run",
                                     "--public-root", plan["public_root"], "--workers", "1", "--domain", "rbr"])
    monkeypatch.setattr(cli.ev, "run", lambda folder: calls.append(folder) or {"status": "test_stub"}, raising=False)
    cli.main()
    assert calls == [case.resolve()]
    assert json.loads(capsys.readouterr().out)["status"] == "test_stub"


def test_public_help_needs_no_model_or_data():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "--action" in result.stdout and "--eval-ids" in result.stdout
    assert "--_phase" not in result.stdout
