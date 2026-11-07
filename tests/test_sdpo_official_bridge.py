import hashlib
import json
from pathlib import Path

import pytest

from eesd.sdpo_official import export_cohort, verify_export, compute_score, validate_upstream


def frozen(root):
    root.mkdir()
    def row(sid, pid, key, prefix, count):
        return dict(source_id=sid, problem_id=pid, task='Echo input', buggy_code='print(0)',
                    **{key: [dict(id=str(i), input=f'{prefix}{i}', expected=f'{prefix}{i}')
                             for i in range(count)]})
    files = {
        'train_public_master64.jsonl': [row('t0', 'p0', 'public_tests', 'PUBLIC', 64)],
        'train_calibration.jsonl': [row('t0', 'p0', 'calibration_tests', 'CAL_SECRET', 8)],
        'train_audit_do_not_train.jsonl': [row('t0', 'p0', 'audit_tests', 'AUDIT_SECRET', 8)],
        'eval_private.jsonl': [row('e0', 'p1', 'evaluation_tests', 'FINAL_SECRET', 15)],
        'meta_do_not_train.jsonl': [row('m0', 'p2', 'evaluation_tests', 'META_SECRET', 80)],
    }
    for name, rows in files.items():
        (root/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    seal(root)
    return root


def seal(root):
    (root/'manifest.json').write_text(json.dumps({
        'schema': 'eesd-rich64-frozen-v1', 'calibration_tests': 8, 'reserved_audit_tests': 8,
        'file_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in root.glob('*.jsonl')}}))


def test_export_preserves_public_bank_and_hides_calibration_meta_final(tmp_path):
    import pyarrow.parquet as pq
    source = frozen(tmp_path/'source')
    out = tmp_path/'export'
    receipt = export_cohort(source, out, train_limit=1, eval_limit=1)
    train = pq.read_table(out/'train.parquet').to_pylist()
    final = pq.read_table(out/'test.parquet').to_pylist()
    assert receipt['status'] == 'data_prepared_not_trained'
    assert receipt['eval_problems'] == 1
    assert all(secret not in json.dumps(train) for secret in
               ['CAL_SECRET', 'AUDIT_SECRET', 'FINAL_SECRET', 'META_SECRET'])
    assert 'PUBLIC' not in json.dumps(train[0]['prompt'])
    assert len(json.loads(train[0]['reward_model']['ground_truth'])['tests']) == 64
    assert len(json.loads(final[0]['reward_model']['ground_truth'])['tests']) == 15
    assert verify_export(out) == receipt
    with pytest.raises(FileExistsError):
        export_cohort(source, out, train_limit=1, eval_limit=1)


def test_export_rejects_overlap_even_after_manifest_is_resealed(tmp_path):
    source = frozen(tmp_path/'source')
    p = source/'eval_private.jsonl'
    p.write_text(p.read_text().replace('p1', 'p0'))
    seal(source)
    with pytest.raises(ValueError, match='overlap'):
        export_cohort(source, tmp_path/'out', train_limit=1, eval_limit=1)


def test_real_dataset_unicode_line_separators_are_not_jsonl_boundaries(tmp_path):
    import pyarrow.parquet as pq
    source = frozen(tmp_path/'source')
    p = source/'train_public_master64.jsonl'
    r = json.loads(p.read_text())
    r['task'] = 'part A\u2028part B\u0085part C'
    p.write_text(json.dumps(r, ensure_ascii=False)+'\n', encoding='utf-8')
    seal(source)
    out = tmp_path/'out'
    export_cohort(source, out, train_limit=1, eval_limit=1)
    assert r['task'] in pq.read_table(out/'train.parquet').to_pylist()[0]['prompt'][1]['content']


def test_export_rejects_missing_hash_and_tampered_data(tmp_path):
    source = frozen(tmp_path/'source')
    with (source/'train_public_master64.jsonl').open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='hash'):
        export_cohort(source, tmp_path/'out', train_limit=1, eval_limit=1)


def test_verify_export_rejects_changed_parquet(tmp_path):
    out = tmp_path/'out'
    export_cohort(frozen(tmp_path/'source'), out, train_limit=1, eval_limit=1)
    with (out/'train.parquet').open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(ValueError, match='hash'):
        verify_export(out)


def payload(role='train'):
    return json.dumps({'schema': 'sdpo-rich64-tests-v1', 'role': role,
                       'tests': [{'id': '0', 'input': '7\n', 'expected': '7\n'}]})


def test_reward_rejects_mismatched_roles_before_execution():
    with pytest.raises(ValueError, match='role'):
        compute_score('runbugrun_rich64', 'print(7)', payload('test'), {'split': 'train'})
    with pytest.raises(ValueError, match='data_source'):
        compute_score('unknown', 'print(7)', payload(), {'split': 'train'})


def test_private_evaluation_returns_no_feedback(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('eesd.sdpo_official.run_tests_batched', lambda code, tests: [
        SimpleNamespace(status='WRONG_OUTPUT', actual='0', stderr='')])
    result = compute_score('runbugrun_rich64', 'print(0)', payload('test'), {'split': 'test'})
    assert result['score'] == 0
    assert result['feedback'] == ''
    assert '7' not in result['pred']


def test_public_reward_is_all_pass_not_fraction(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('eesd.sdpo_official.run_tests_batched', lambda code, tests: [
        SimpleNamespace(status='PASS', actual='7', stderr=''),
        SimpleNamespace(status='WRONG_OUTPUT', actual='0', stderr='')])
    monkeypatch.setattr('eesd.sdpo_official._upstream_feedback', lambda records, truncated: 'public failure')
    data = json.loads(payload())
    data['tests'].append({'id':'1', 'input':'8', 'expected':'8'})
    result = compute_score('runbugrun_rich64', 'print(7)', json.dumps(data), {'split':'train'})
    assert result['score'] == 0
    assert result['acc'] == .5
    assert result['feedback'] == 'public failure'


def test_infrastructure_failure_is_not_candidate_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('bubblewrap unavailable')
    monkeypatch.setattr('eesd.sdpo_official.run_tests_batched', fail)
    with pytest.raises(RuntimeError, match='bubblewrap'):
        compute_score('runbugrun_rich64', 'print(7)', payload(), {'split':'train'})


def test_fake_upstream_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='upstream'):
        validate_upstream(tmp_path)


def test_report_uses_whole_problem_score_and_checks_complete_cohort(tmp_path):
    import pyarrow.parquet as pq
    from eesd.sdpo_official import summarize_validation
    out = tmp_path/'out'
    export_cohort(frozen(tmp_path/'source'), out, train_limit=1, eval_limit=1)
    test = pq.read_table(out/'test.parquet').to_pylist()[0]
    row = {'gts': test['reward_model']['ground_truth'], 'score': 0., 'acc': 14/15,
           'test_count': 15, 'passed_tests': 14, 'feedback': '', 'step': 5}
    path = tmp_path/'5.jsonl'
    path.write_text(json.dumps(row)+'\n')
    report = summarize_validation(path, out)
    assert report['problem_accuracy'] == 0.
    assert report['mean_test_fraction_diagnostic'] == 14/15
    path.write_text('')
    with pytest.raises(ValueError, match='count'):
        summarize_validation(path, out)


def test_report_rejects_wrong_tests_or_nonbinary_scores(tmp_path):
    import pyarrow.parquet as pq
    from eesd.sdpo_official import summarize_validation
    out = tmp_path/'out'
    export_cohort(frozen(tmp_path/'source'), out, train_limit=1, eval_limit=1)
    test = pq.read_table(out/'test.parquet').to_pylist()[0]
    path = tmp_path/'5.jsonl'
    row = {'gts': test['reward_model']['ground_truth'], 'score': .5,
           'acc': .5, 'test_count': 15, 'passed_tests': 7, 'feedback': '', 'step': 5}
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='binary'):
        summarize_validation(path, out)
    row['score'] = 0
    row['gts'] = '{}'
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='cohort'):
        summarize_validation(path, out)
