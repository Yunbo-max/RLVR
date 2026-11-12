import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_latest_cli_rejects_removed_simplified_sdpo(tmp_path):
    p = subprocess.run([sys.executable, str(ROOT/'scripts/run_t64_grouped_five_round.py'),
        '--data-dir', str(tmp_path), '--output', str(tmp_path/'out'),
        '--init-adapter', str(tmp_path), '--arm', 'sdpo', '--dry-run'],
        capture_output=True, text=True)
    assert p.returncode == 2
    assert 'invalid choice' in p.stderr
    assert not (tmp_path/'out').exists()


@pytest.mark.skipif(shutil.which('bash') is None, reason='Linux matrix dispatch')
@pytest.mark.parametrize('stage,train,eval_count,g', [('pilot',64,64,4), ('mechanism',512,199,8), ('scale',8000,199,8)])
def test_t64_matrix_replaces_old_arm_with_official_preflight_and_run(tmp_path, stage, train, eval_count, g):
    commands = _record_matrix(tmp_path, stage, 64)
    calls = [c for c in commands if c[0].endswith('run_sdpo_official.py')]
    assert [c[1] for c in calls] == ['check', 'run']
    for c in calls:
        assert c[c.index('--rollouts')+1] == str(g)
        assert c[c.index('--seed')+1] == '1701'
    exports = [c for c in commands if c[0].endswith('export_sdpo_rich64.py')]
    assert len(exports) == 1
    assert exports[0][exports[0].index('--train-limit')+1] == str(train)
    assert exports[0][exports[0].index('--eval-limit')+1] == str(eval_count)
    internal = [c for c in commands if c[0].endswith('run_t64_grouped_five_round.py')]
    assert all(c[c.index('--arm')+1] != 'sdpo' for c in internal)
    assert commands.index(calls[0]) < commands.index(internal[0])
    assert commands.index(calls[1]) > commands.index(internal[-1])
    summaries = [c for c in commands if c[0].endswith('summarize_sdpo_official.py')]
    assert len(summaries) == 1
    assert summaries[0][summaries[0].index('--validation')+1].endswith(f'/{5*train}.jsonl')


@pytest.mark.skipif(shutil.which('bash') is None, reason='Linux matrix dispatch')
def test_t4_sweep_does_not_mislabel_full_t64_sdpo_as_t4(tmp_path):
    commands = _record_matrix(tmp_path, 'pilot', 4)
    assert not any(c[0].endswith('run_sdpo_official.py') for c in commands)
    internal = [c for c in commands if c[0].endswith('run_t64_grouped_five_round.py')]
    assert all(c[c.index('--arm')+1] != 'sdpo' for c in internal)


def _record_matrix(tmp_path, stage, t):
    # Record the real shell's subprocess boundaries; never start GPU workers.
    bin_dir = tmp_path/'bin'
    bin_dir.mkdir()
    recorder = bin_dir/'python'
    recorder.write_text(f'#!{sys.executable}\n'+'''import json, os, sys
from pathlib import Path
a=sys.argv[1:]
with open(os.environ['CALL_LOG'],'a') as f: f.write(json.dumps(a)+'\\n')
if a[0].endswith('run_sdpo_official.py') and a[1]=='run':
    Path(a[a.index('--output')+1]).mkdir(parents=True)
''')
    recorder.chmod(0o755)
    init = tmp_path/'init'
    init.mkdir()
    (init/'adapter_config.json').write_text('{}')
    log = tmp_path/'calls.jsonl'
    env = dict(os.environ, PATH=str(bin_dir)+os.pathsep+os.environ['PATH'],
               SDPO_PYTHON=str(recorder), CALL_LOG=str(log))
    result = subprocess.run(['bash', str(ROOT/'scripts/run_t64_group_matrix.sh'),
        str(tmp_path/'data'), str(tmp_path/'out'), str(t), stage, '1701', str(init)],
        cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return [json.loads(s) for s in log.read_text().splitlines()]
