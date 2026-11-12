"""Data and sandbox bridge to the unmodified authors' SDPO implementation.

This module does not implement an SDPO loss or teacher update. Those run inside
the pinned upstream trainer. RunBugRun prompt/execution semantics are an
explicit benchmark adaptation, not a reproduction of the paper's LCB scores.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

from eesd.strict_execution import run_tests_batched

UPSTREAM_URL = 'https://github.com/lasgroup/SDPO.git'
UPSTREAM_COMMIT = '7c457fc1b1f636ae794eb0362ba37d4743b06fbc'
MODEL_ID = 'Qwen/Qwen2.5-Coder-7B-Instruct'
MODEL_REVISION = 'c03e6d358207e414f1eca0bb1891e29f1db0e242'
SOURCE_FILES = ('train_public_master64.jsonl', 'train_calibration.jsonl',
                'train_audit_do_not_train.jsonl', 'eval_private.jsonl',
                'meta_do_not_train.jsonl')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def validate_upstream(root: Path) -> dict:
    root = root.resolve()
    try:
        head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                       text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD', '--'],
                               capture_output=True).returncode
    except (OSError, subprocess.CalledProcessError) as e:
        raise ValueError('invalid upstream checkout') from e
    if head != UPSTREAM_COMMIT or dirty:
        raise ValueError('upstream must be the pinned, unmodified SDPO checkout')
    paths = ['verl/trainer/ppo/core_algos.py', 'verl/trainer/ppo/ray_trainer.py',
             'verl/workers/actor/dp_actor.py', 'verl/trainer/config/actor/actor.yaml',
             'experiments/rich_feedback/run_sdpo.sh', 'verl/utils/reward_score/feedback/code.py']
    return {'repository': UPSTREAM_URL, 'commit': head,
            'files': {name: sha256(root/name) for name in paths}}


def _read(path: Path) -> list[dict]:
    # Unicode U+2028/U+0085 can occur inside real problem text; only LF is a
    # JSONL record boundary, unlike str.splitlines().
    with path.open(encoding='utf-8') as stream:
        return [json.loads(s) for s in stream if s.strip()]


def _tests(rows: list[dict], key: str, count: int | None = None):
    ids = [str(r['source_id']) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate source_id')
    for row in rows:
        tests = row[key]
        if not tests or (count is not None and len(tests) != count):
            raise ValueError(f'invalid {key} test count')
        if any(not isinstance(t.get(k), str) for t in tests for k in ('input', 'expected')):
            raise ValueError('tests must have string input and expected output')
        if len({t['input'] for t in tests}) != len(tests):
            raise ValueError('duplicate test inputs')


def _prompt(row: dict) -> list[dict]:
    # Byte-identical to the latest Rich64 no-feedback repair prompt.
    return [
        {'role': 'system', 'content': 'You are an expert Python program repair model.'},
        {'role': 'user', 'content':
         'Repair the following buggy Python stdin/stdout program. Return a '
         'complete corrected Python script only, without markdown. '
         'Do not add examples, prose or test harness.\n\n'
         'Problem:\n' + row['task'] + '\n\nBuggy code:\n' + row['buggy_code']},
    ]


def _convert(row: dict, role: str) -> dict:
    key = 'public_tests' if role == 'train' else 'evaluation_tests'
    tests = [{k: str(t[k]) for k in ('id', 'input', 'expected')} for t in row[key]]
    return {'data_source': 'runbugrun_rich64', 'prompt': _prompt(row), 'ability': 'code',
            'reward_model': {'style': 'rule', 'ground_truth': json.dumps({
                'schema': 'sdpo-rich64-tests-v1', 'role': role, 'tests': tests}, ensure_ascii=False)},
            'extra_info': {'split': role, 'index': str(row['source_id']),
                           'problem_id': str(row['problem_id'])}}


def export_cohort(source: Path, output: Path, *, train_limit: int, eval_limit: int) -> dict:
    """Reuse frozen ordered rows. Never import gold code or hidden training tests."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source/'manifest.json').read_text())
    if manifest.get('schema') != 'eesd-rich64-frozen-v1':
        raise ValueError('expected Rich64 frozen manifest')
    for name in SOURCE_FILES:
        if manifest.get('file_sha256', {}).get(name) != sha256(source/name):
            raise ValueError(f'source hash mismatch: {name}')
    train = _read(source/SOURCE_FILES[0])
    calibration = _read(source/SOURCE_FILES[1])
    audit = _read(source/SOURCE_FILES[2])
    final = _read(source/SOURCE_FILES[3])
    meta = _read(source/SOURCE_FILES[4])
    _tests(train, 'public_tests', 64)
    _tests(calibration, 'calibration_tests', 8)
    _tests(audit, 'audit_tests', 8)
    _tests(final, 'evaluation_tests')
    groups = [{str(r['problem_id']) for r in rows} for rows in (train, final, meta)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError('train/evaluation/meta problem overlap')
    if len(final) != len(groups[1]):
        raise ValueError('evaluation problem IDs must be unique')
    bycal = {r['source_id']: r for r in calibration}
    byaudit = {r['source_id']: r for r in audit}
    for r in train:
        c, a = bycal.get(r['source_id']), byaudit.get(r['source_id'])
        if c is None or a is None or c['problem_id'] != r['problem_id'] or a['problem_id'] != r['problem_id']:
            raise ValueError('missing or mismatched calibration/audit rows')
        banks = [{t['input'] for t in z[k]} for z, k in
                 ((r, 'public_tests'), (c, 'calibration_tests'), (a, 'audit_tests'))]
        if any(banks[i] & banks[j] for i in range(3) for j in range(i+1, 3)):
            raise ValueError('public/calibration/audit input overlap')
    if not 0 < train_limit <= len(train) or not 0 < eval_limit <= len(final):
        raise ValueError('requested cohort exceeds frozen rows')
    train, final = train[:train_limit], final[:eval_limit]
    output.mkdir(parents=True, exist_ok=False)
    for name, rows, role in [('train.parquet', train, 'train'), ('test.parquet', final, 'test')]:
        pq.write_table(pa.Table.from_pylist([_convert(r, role) for r in rows]), output/name,
                       compression='zstd', row_group_size=32)
    receipt = {'schema': 'sdpo-rich64-export-v1', 'status': 'data_prepared_not_trained',
               'source_manifest_sha256': sha256(source/'manifest.json'),
               'source_directory': str(source.resolve()),
               'train_bugs': len(train), 'train_problems': len({r['problem_id'] for r in train}),
               'eval_problems': len(final),
               'train_source_ids': [r['source_id'] for r in train],
               'eval_source_ids': [r['source_id'] for r in final],
               'training_tests': '64 public tests only; C8/A8/meta never serialized',
               'evaluation_tests': 'entire frozen official test bank per selected problem',
               'files': {n: sha256(output/n) for n in ('train.parquet', 'test.parquet')}}
    (output/'manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def verify_export(folder: Path) -> dict:
    receipt = json.loads((folder/'manifest.json').read_text())
    if receipt.get('schema') != 'sdpo-rich64-export-v1':
        raise ValueError('expected SDPO export manifest')
    if set(receipt.get('files', {})) != {'train.parquet', 'test.parquet'}:
        raise ValueError('missing parquet hash')
    for name, digest in receipt['files'].items():
        if sha256(folder/name) != digest:
            raise ValueError(f'export hash mismatch: {name}')
    return receipt


def summarize_validation(path: Path, data_dir: Path) -> dict:
    """Report binary whole-program accuracy, never upstream's fractional `acc`."""
    import pyarrow.parquet as pq
    receipt = verify_export(data_dir)
    expected = pq.read_table(data_dir/'test.parquet').to_pylist()
    rows = _read(path)
    if len(rows) != len(expected) or len(rows) != receipt['eval_problems']:
        raise ValueError('evaluation count differs from frozen cohort')
    for result, task in zip(rows, expected):
        if result['gts'] != task['reward_model']['ground_truth']:
            raise ValueError('evaluation cohort/order mismatch')
        if result['score'] not in (0, 1):
            raise ValueError('expected binary whole-program scores')
        n = len(json.loads(result['gts'])['tests'])
        if (result['test_count'] != n or not 0 <= result['passed_tests'] <= n
            or result['score'] != int(result['passed_tests'] == n)
            or result.get('feedback', '') != ''):
            raise ValueError('inconsistent test outcome or private feedback')
    if len({r['step'] for r in rows}) != 1:
        raise ValueError('mixed checkpoints in validation file')
    return {'schema': 'sdpo-rich64-evaluation-v1', 'step': rows[0]['step'],
            'eval_problems': len(rows), 'solved_problems': sum(r['score'] for r in rows),
            'problem_accuracy': sum(r['score'] for r in rows)/len(rows),
            'mean_test_fraction_diagnostic': sum(r['acc'] for r in rows)/len(rows),
            'eval_source_ids': receipt['eval_source_ids'],
            'solved_vector': [int(r['score']) for r in rows],
            'data_manifest_sha256': sha256(data_dir/'manifest.json'),
            'validation_sha256': sha256(path)}


@lru_cache(maxsize=1)
def _official_feedback_module():
    root = Path(os.environ['SDPO_UPSTREAM_ROOT'])
    validate_upstream(root)
    spec = importlib.util.spec_from_file_location('sdpo_author_feedback', root/'verl/utils/reward_score/feedback/code.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _upstream_feedback(records: list[dict], truncated: bool) -> str:
    # Only render feedback here: upstream's soft execution is not invoked.
    return _official_feedback_module().format_test_feedback(records, was_truncated=truncated)


def compute_score(data_source: str, solution_str: str, ground_truth: str,
                  extra_info: dict | None = None) -> dict:
    """VERL reward-hook signature. A sandbox failure aborts instead of scoring zero."""
    if data_source != 'runbugrun_rich64':
        raise ValueError('unknown data_source')
    data = json.loads(ground_truth)
    extra_info = extra_info or {}
    role = data.get('role')
    if data.get('schema') != 'sdpo-rich64-tests-v1' or role not in ('train', 'test') or role != extra_info.get('split'):
        raise ValueError('test role/schema mismatch')
    tests = data['tests']
    _tests([{'source_id': 'callback', 'tests': tests}], 'tests')
    # Repair tasks explicitly request a bare stdin/stdout script. Accept a fenced
    # answer as the historical Rich64 extractor does, but never execute prose.
    code = solution_str.strip()
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', code, re.DOTALL | re.IGNORECASE)
    if blocks:
        code = blocks[0].strip()
    outcomes = run_tests_batched(code, tests)
    if len(outcomes) != len(tests):
        raise RuntimeError('sandbox returned incomplete test outcomes')
    passes = sum(v.status == 'PASS' for v in outcomes)
    records = []
    for i, (t, v) in enumerate(zip(tests, outcomes)):
        actual = v.actual
        if v.status == 'TIMEOUT':
            actual = 'Time out'
        elif v.status in ('RUNTIME_EXCEPTION', 'COMPILE_ERROR'):
            actual = 'Error: ' + v.stderr
        records.append({'test_idx': i, 'input': t['input'], 'expected': t['expected'],
                        'actual': actual, 'passed': v.status == 'PASS', 'debug': ''})
    truncated = bool(extra_info.get('truncated', False))
    return {'score': float(passes == len(tests)), 'acc': passes/len(tests),
            'feedback': _upstream_feedback(records, truncated) if role == 'train' else '',
            'pred': '', 'test_count': len(tests), 'passed_tests': passes,
            'truncated': int(truncated),
            'timed_out': int(any(v.status == 'TIMEOUT' for v in outcomes))}
