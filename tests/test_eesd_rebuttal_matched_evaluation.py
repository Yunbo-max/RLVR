"""CPU contracts for the prospective protocol; no mock is a scientific outcome."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest
import yaml

from eesd import rebuttal_eval as ev

ROOT = Path(__file__).resolve().parents[1]
REAL_PBPF = ROOT.parent / 'pbpf-original'
REAL_PIN = ev.PBPF_PIN
PBPF = REAL_PBPF


@pytest.fixture(scope='session')
def stub_checkout(tmp_path_factory):
    """Test-only source/pin injection; never an evaluation evidence fixture."""
    root = tmp_path_factory.mktemp('pbpf-schema-stub')
    for name in ev.PBPF_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Schema fixture only; no execution implementation.\n')
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    subprocess.run(['git', 'add', '.'], cwd=root, check=True)
    subprocess.run(['git', '-c', 'user.name=Contract test', '-c', 'user.email=contract@example.invalid', 'commit', '-qm', 'test-only source pin'], cwd=root, check=True)
    pin = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    return root, pin


@pytest.fixture(autouse=True)
def isolated_test_pin(monkeypatch, stub_checkout):
    root, pin = stub_checkout
    monkeypatch.setattr(ev, 'PBPF_PIN', pin)
    monkeypatch.setattr(sys.modules[__name__], 'PBPF', root)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def fixture(tmp_path, dataset='runbugrun', family='qwen25_7b', baseline=False):
    cell = dataset + '__' + family
    profile = next(c for c in json.loads((ROOT / 'configs/rebuttal/model_dataset_matrix.json').read_text())['cells'] if c['cell'] == cell)
    model_config = ROOT / profile['model_config']
    config = yaml.safe_load(model_config.read_text())
    case = tmp_path / cell / ('C00' if baseline else 'C03')
    case.mkdir(parents=True)
    adapter = case / ('initial' if baseline else 'model/adapter')
    adapter.mkdir(parents=True)
    put(adapter / 'adapter_config.json', dict(peft_type='LORA', task_type='CAUSAL_LM', base_model_name_or_path=config['model_id'], revision=config['revision'], r=1, lora_alpha=2, lora_dropout=0.0, target_modules=['model.language_model.layers.0.self_attn.q_proj'] if family == 'gemma3_4b' else 'all-linear'))
    header = json.dumps({'layer.lora_A.weight': {'dtype': 'F32', 'shape': [1, 1], 'data_offsets': [0, 4]}, 'layer.lora_B.weight': {'dtype': 'F32', 'shape': [1, 1], 'data_offsets': [4, 8]}}).encode()
    (adapter / 'adapter_model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * 8)
    public = tmp_path / cell / 'bundle' / profile['domain'] / 'public'
    hidden = public.parent / 'evaluator'
    replay = dataset.endswith('_replay')
    pubs, privs = [], []
    for i in range(500):
        identity = dict(task_id=f'{profile["domain"]}/{i}', source_component_id=f'{profile["domain"]}-source-{i}', split='primary')
        tests = [dict(id=str(j), input=f'{j}\r\n', **({'output': f'{j}\n'} if replay else {'expected': f'{j}\n'})) for j in range(10)]
        if replay:
            identity['domain'] = profile['domain']
            for t in tests:
                t.update(pair_sha256=ev.pair_sha(t['input'], t['output']), original_pools=['apps_input_output' if dataset == 'apps_replay' else 'private_tests'], known_exposed_input=False)
            pubs.append({**identity, 'statement': 'Print the input. α\u0085β\n雪', 'visible_tests': [{k: t[k] for k in ('input', 'output')} for t in tests[:4]]})
            privs.append({**identity, 'statement': pubs[-1]['statement'], 'tests': tests, 'known_exposed_input_sha256': []})
        else:
            identity['protocol'] = 'RBR-generated-repair' if dataset == 'runbugrun' else 'CodeARC-Replay'
            if dataset == 'runbugrun': identity['base_bug_id'] = f'bug-{i}'
            for j, t in enumerate(tests):
                t['hidden'] = j >= 4
                if dataset == 'codearc': t['expected_error'] = False
                else: t['original_id'] = f'original-{j}'
            visible = [{k: t[k] for k in (('id', 'input', 'expected') if dataset == 'runbugrun' else ('id', 'input', 'expected', 'expected_error'))} for t in tests[:4]]
            pubs.append({**identity, 'task_text': 'Print the input. α\u0085β\n雪', 'visible_tests': visible, **({'buggy_code': 'print(0)'} if dataset == 'runbugrun' else {})})
            privs.append({**identity, 'reference_code': 'print(input())', 'tests': tests})
    jsonl(public / 'tasks.jsonl', pubs)
    jsonl(hidden / 'tasks.jsonl', privs)
    if replay:
        for folder, schema in [(public, 'eesd-replay-public-v1'), (hidden, 'eesd-replay-evaluator-v1')]:
            put(folder / 'manifest.json', dict(schema=schema, domain=profile['domain'], task_kind='stdin_synthesis', counts={'primary': 500}, tasks={'sha256': ev.file_sha(folder / 'tasks.jsonl'), 'bytes': (folder / 'tasks.jsonl').stat().st_size}))
        bundle = public.parent.parent
        files = [public / 'tasks.jsonl', public / 'manifest.json', hidden / 'tasks.jsonl', hidden / 'manifest.json']
        put(bundle / 'admission.json', dict(schema='eesd-replay-admission-v1', status='data-admitted-not-generated-not-scored', outputs={p.relative_to(bundle).as_posix(): {'sha256': ev.file_sha(p), 'bytes': p.stat().st_size} for p in files}))
    else:
        schema = 'apbpf-rbr-generated-materialization-v1' if dataset == 'runbugrun' else 'apbpf-codearc-replay-materialization-v1'
        common = dict(schema=schema, public_tasks_sha256=ev.file_sha(public / 'tasks.jsonl'))
        put(public / 'manifest.json', common)
        put(hidden / 'manifest.json', {**common, 'evaluator_tasks_sha256': ev.file_sha(hidden / 'tasks.jsonl')})
    ids = case / 'eval-ids.jsonl'
    jsonl(ids, [{'source_component_id': r['source_component_id']} for r in pubs])
    corrections = case / 'corrections.jsonl'
    jsonl(corrections, [dict(source_component_id='train-a', split='train', model_id=config['model_id'], model_revision=config['revision'], domain=profile['domain']), dict(source_component_id='dev-a', split='development', model_id=config['model_id'], model_revision=config['revision'], domain=profile['domain'])])
    provenance = dict(correction_file_sha256=ev.file_sha(corrections), provenance_status='unverified_problem_id_disjointness', errors=[], evaluation=dict(eval_input_sha256=ev.file_sha(ids), eval_rows=500, source_level_overlap_checked=True, problem_level_overlap_checked=False))
    put(case / 'data_provenance.json', provenance)
    setup = dict(model_id=config['model_id'], revision=config['revision'], model_config_path=str(model_config), model_config_sha256=ev.file_sha(model_config), corrections_path=str(corrections), corrections_sha256=ev.file_sha(corrections), shared_init_adapter_path=str(adapter), shared_init_adapter_tree_sha256=ev.tree_sha(adapter))
    receipt = dict(schema='eesd-iclr2027-independent-case-v1', case={'id': 'C00' if baseline else 'C03', 'kind': 'baseline' if baseline else 'train'}, study='fixture-unit-contract', seed=1701, status='completed_diagnostic_without_gpu' if baseline else 'trained_awaiting_independent_evaluation', original_paper_cell=cell, setup=setup, data_provenance_sha256=ev.file_sha(case / 'data_provenance.json'), frozen_case_input_hashes={'eval_ids_sha256': ev.file_sha(ids)}, trained_adapter_tree_sha256=ev.tree_sha(adapter), cell_identity_audit=dict(status='verified_model_and_dataset_identity', cell=cell, model_id=config['model_id'], model_revision=config['revision'], dataset=dataset, source_sha256=ev.file_sha(corrections)))
    put(case / 'case.json', receipt)
    return dict(case=case, pbpf=PBPF, public=public, evaluator=hidden, eval_ids=ids, workers=1)


def reseal(f):
    pub = json.loads((f['public'] / 'manifest.json').read_text())
    hid = json.loads((f['evaluator'] / 'manifest.json').read_text())
    pub['public_tasks_sha256'] = ev.file_sha(f['public'] / 'tasks.jsonl')
    hid.update(public_tasks_sha256=pub['public_tasks_sha256'], evaluator_tasks_sha256=ev.file_sha(f['evaluator'] / 'tasks.jsonl'))
    put(f['public'] / 'manifest.json', pub)
    put(f['evaluator'] / 'manifest.json', hid)


@pytest.mark.parametrize('dataset', ['runbugrun', 'codearc', 'apps_replay', 'codecontests_replay'])
@pytest.mark.parametrize('family', ['qwen25_7b', 'deepseek_6p7b', 'gemma3_4b'])
def test_prepare_all_registered_cells_without_ml_imports(tmp_path, dataset, family):
    f = fixture(tmp_path, dataset, family)
    plan = ev.prepare(**f)
    assert plan['schema'] == 'eesd-rebuttal-per-case-matched-evaluator-v1'
    assert plan['evaluation_protocol'] == plan['prospective_protocol'] == 'eesd-rebuttal-matched-prospective-v1'
    assert plan['evaluation_sources'] == 500
    assert plan['task_kind'] == ('replay_synthesis' if dataset.endswith('_replay') else 'repair')
    assert plan['generation_policy']['max_new_tokens'] == (512 if dataset == 'codearc' else 1024)
    assert plan['generation_policy']['quantization']['bnb_4bit_quant_type'] == 'nf4'
    assert not (f['case'] / 'evaluation/complete.json').exists()
    ev.check_frozen(plan, f['case'])


def test_frozen_baseline_uses_initial_adapter(tmp_path):
    f = fixture(tmp_path, baseline=True)
    plan = ev.prepare(**f)
    assert plan['adapter_path'].endswith('/initial')


@pytest.mark.parametrize('change', ['domain', 'model', 'adapter', 'mapping', 'cell'])
def test_wrong_identity_rejected_before_output(tmp_path, change):
    f = fixture(tmp_path)
    if change == 'domain': f['domain'] = 'codearc'
    elif change == 'adapter': (f['case'] / 'model/adapter/adapter_model.safetensors').write_bytes(b'broken')
    elif change == 'mapping': jsonl(f['eval_ids'], [{'source_component_id': 'invented'}])
    else:
        r = json.loads((f['case'] / 'case.json').read_text())
        if change == 'model': r['setup']['model_id'] = 'google/gemma-3-4b-it'
        else: r['original_paper_cell'] = 'codearc__qwen25_7b'
        put(f['case'] / 'case.json', r)
    with pytest.raises(ValueError): ev.prepare(**f)
    assert not (f['case'] / 'evaluation').exists()


@pytest.mark.parametrize('change', ['hidden_identity', 'public_leak', 'duplicate_task', 'duplicate_test', 'empty_tests', 'visible_mismatch'])
def test_invalid_materializations_fail_closed(tmp_path, change):
    f = fixture(tmp_path)
    p = ev.read_jsonl(f['public'] / 'tasks.jsonl')
    h = ev.read_jsonl(f['evaluator'] / 'tasks.jsonl')
    if change == 'hidden_identity': h[0]['base_bug_id'] = 'wrong'
    elif change == 'public_leak': p[0]['reference_code'] = 'print(input())'
    elif change == 'duplicate_task': p.append(p[0]); h.append(h[0])
    elif change == 'duplicate_test': h[0]['tests'][5] = h[0]['tests'][4]
    elif change == 'empty_tests': h[0]['tests'] = []
    elif change == 'visible_mismatch': h[0]['tests'][0]['input'] = 'changed'
    jsonl(f['public'] / 'tasks.jsonl', p)
    jsonl(f['evaluator'] / 'tasks.jsonl', h)
    reseal(f)
    with pytest.raises(ValueError): ev.prepare(**f)


@pytest.mark.parametrize('target', ['case.json', 'data_provenance.json', 'public', 'adapter', 'mapping'])
def test_changed_frozen_inputs_are_fatal(tmp_path, target):
    f = fixture(tmp_path)
    plan = ev.prepare(**f)
    path = {'public': f['public'] / 'tasks.jsonl', 'adapter': f['case'] / 'model/adapter/adapter_model.safetensors', 'mapping': f['eval_ids']}.get(target, f['case'] / target)
    with path.open('ab') as stream: stream.write(b' ')
    with pytest.raises(ValueError): ev.check_frozen(plan, f['case'])


def test_unicode_jsonl_and_original_newlines_preserved(tmp_path):
    f = fixture(tmp_path, 'apps_replay')
    rows = ev.read_jsonl(f['public'] / 'tasks.jsonl')
    assert 'α\u0085β\n雪' in rows[0]['statement']
    assert rows[0]['visible_tests'][0]['input'] == '0\r\n'
    assert rows[0]['visible_tests'][0]['output'] == '0\n'
    ev.prepare(**f)


def test_duplicate_eval_ids_rejected_even_when_refrozen(tmp_path):
    f = fixture(tmp_path)
    rows = ev.read_jsonl(f['eval_ids'])
    rows[-1] = rows[0]
    jsonl(f['eval_ids'], rows)
    r = json.loads((f['case'] / 'case.json').read_text())
    r['frozen_case_input_hashes']['eval_ids_sha256'] = ev.file_sha(f['eval_ids'])
    put(f['case'] / 'case.json', r)
    with pytest.raises(ValueError, match='duplicate|source-ID'): ev.prepare(**f)


def test_infrastructure_probe_happens_before_generation(tmp_path, monkeypatch):
    f = fixture(tmp_path)
    ev.prepare(**f)
    called = []
    def failed_probe(*args, **kwargs):
        raise RuntimeError('bwrap: namespace unavailable')
    monkeypatch.setattr(ev, 'probe_sandbox', failed_probe)
    monkeypatch.setattr(ev, 'run_stage', lambda *args, **kwargs: called.append(args))
    with pytest.raises(RuntimeError, match='namespace'): ev.run(f['case'])
    assert called == []
    assert not (f['case'] / 'evaluation/complete.json').exists()
    assert json.loads((f['case'] / 'evaluation/plan.json').read_text())['status'] == 'failed'


def test_help_does_not_import_ml_stack():
    result = subprocess.run([sys.executable, '-c', "import runpy,sys;sys.argv=['runner','--help'];\ntry: runpy.run_path('scripts/run_eesd_rebuttal_matched_evaluation.py',run_name='__main__')\nexcept SystemExit as e:\n assert e.code == 0\n assert not any(k in sys.modules for k in ('torch','transformers','peft','bitsandbytes'))"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('dataset', ['runbugrun', 'codearc'])
def test_original_repeated_inputs_keep_distinct_test_ids(tmp_path, dataset):
    f = fixture(tmp_path, dataset)
    hidden = ev.read_jsonl(f['evaluator'] / 'tasks.jsonl')
    hidden[0]['tests'][5]['input'] = hidden[0]['tests'][4]['input']
    jsonl(f['evaluator'] / 'tasks.jsonl', hidden)
    reseal(f)
    assert ev.prepare(**f)['evaluation_sources'] == 500


def replace_corrections(f, rows):
    receipt = ev.read_json(f['case'] / 'case.json')
    path = Path(receipt['setup']['corrections_path'])
    jsonl(path, rows)
    digest = ev.file_sha(path)
    receipt['setup']['corrections_sha256'] = digest
    receipt['cell_identity_audit']['source_sha256'] = digest
    provenance = ev.read_json(f['case'] / 'data_provenance.json')
    provenance['correction_file_sha256'] = digest
    put(f['case'] / 'data_provenance.json', provenance)
    receipt['data_provenance_sha256'] = ev.file_sha(f['case'] / 'data_provenance.json')
    put(f['case'] / 'case.json', receipt)


@pytest.mark.parametrize('field,value', [('dataset', 'rbr'), ('source_dataset', 'runbugrun'), ('domain', 'runbugrun')])
def test_correction_aliases_match_existing_strict_identity_audit(tmp_path, field, value):
    f = fixture(tmp_path)
    receipt = ev.read_json(f['case'] / 'case.json')
    rows = ev.read_jsonl(receipt['setup']['corrections_path'])
    for row in rows:
        row.pop('domain', None)
        row[field] = value
    replace_corrections(f, rows)
    ev.prepare(**f)


def test_conflicting_source_dataset_is_rejected(tmp_path):
    f = fixture(tmp_path)
    receipt = ev.read_json(f['case'] / 'case.json')
    rows = ev.read_jsonl(receipt['setup']['corrections_path'])
    rows[0]['source_dataset'] = 'codearc'
    replace_corrections(f, rows)
    with pytest.raises(ValueError, match='dataset|domain'):
        ev.prepare(**f)


class CharacterTokenizer:
    """A small tokenizer contract double, never used for scientific generation."""
    def __init__(self):
        self.messages = None
        self.special_token_requests = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True)
        self.messages = messages
        return json.dumps(messages, ensure_ascii=False)

    def encode(self, value, *, add_special_tokens):
        self.special_token_requests.append(add_special_tokens)
        return list(value.encode('utf-8'))


def test_gemma_folding_keeps_instructions_and_unicode_exact():
    tokenizer = CharacterTokenizer()
    wrapper = ev._PublicTokenizer(tokenizer, 'gemma3_4b')
    messages = [{'role': 'system', 'content': 'Complete code only. α\u0085β'},
                {'role': 'user', 'content': '雪\r\nInput preserved.'}]
    prompt = wrapper.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    wrapper.encode(prompt, add_special_tokens=True)
    assert tokenizer.messages == [{'role': 'user', 'content': messages[0]['content'] + '\n\n' + messages[1]['content']}]
    assert tokenizer.special_token_requests == [False]
    assert messages[0]['role'] == 'system'


@pytest.mark.parametrize('dataset', ['runbugrun', 'codearc', 'apps_replay', 'codecontests_replay'])
def test_public_prompt_rejects_hidden_fields_before_loading_any_backend(tmp_path, monkeypatch, dataset):
    f = fixture(tmp_path, dataset)
    row = ev.read_jsonl(f['public'] / 'tasks.jsonl')[0]
    row['tests'] = [{'input': 'secret'}]
    monkeypatch.setattr(ev, '_pbpf_module', lambda *args: pytest.fail('must reject before loading a backend'))
    with pytest.raises(ValueError, match='public'):
        ev.public_prompt(CharacterTokenizer(), row, dict(domain=ev.DATASETS[dataset], family='qwen25_7b'))


@pytest.mark.skipif(not (REAL_PBPF / 'src/pbpf/eesd/replay_prompt.py').is_file(), reason='optional pinned PBPF checkout absent')
@pytest.mark.parametrize('family', ['qwen25_7b', 'deepseek_6p7b', 'gemma3_4b'])
@pytest.mark.parametrize('dataset', ['runbugrun', 'codearc', 'apps_replay', 'codecontests_replay'])
def test_pinned_public_prompt_contract(tmp_path, monkeypatch, family, dataset):
    monkeypatch.setattr(ev, 'PBPF_PIN', REAL_PIN)
    ev.validate_checkout(REAL_PBPF)
    f = fixture(tmp_path, dataset, family)
    row = ev.read_jsonl(f['public'] / 'tasks.jsonl')[0]
    domain = ev.DATASETS[dataset]
    plan = dict(domain=domain, family=family, pbpf_root=str(REAL_PBPF), generation_policy=ev.generation_policy(domain))
    tokenizer = CharacterTokenizer()
    prompt, ids, audit = ev.public_prompt(tokenizer, row, plan)
    assert 0 < len(ids) <= 4096
    assert audit['public_example_ids'] == ['0', '1', '2', '3']
    assert set(tokenizer.special_token_requests) == {False}
    assert tokenizer.messages[0]['role'] == ('user' if family == 'gemma3_4b' or domain.endswith('_replay') else 'system')
    if domain.endswith('_replay'):
        content = tokenizer.messages[0]['content']
        assert row['statement'] in content
        assert json.dumps(row['visible_tests'][0]['input'], ensure_ascii=False) in content
        assert '0\r\n' == row['visible_tests'][0]['input']


def bank_fixture(f):
    plan = ev.prepare(**f)
    receipt = dict(generator_dependency_versions={name: 'contract-double' for name in ('torch', 'transformers', 'peft', 'bitsandbytes')},
                   generation_runtime={'test_only_contract_double': True})
    records = [dict(task, completion='print(input())', program='print(input())', input_tokens=20,
                    output_tokens=4, prompt_sha256='a' * 64, cutoff=False) for task in plan['primary_tasks']]
    return plan, receipt, records


@pytest.mark.parametrize('target', ['000000.json', 'run.json', 'complete.json', 'extra.json'])
def test_sealed_bank_rejects_changed_or_extra_records(tmp_path, target):
    f = fixture(tmp_path)
    plan, receipt, records = bank_fixture(f)
    ev.seal_bank(plan, receipt, records)
    assert len(ev.validate_bank(plan)[1]) == 500
    path = f['case'] / 'evaluation/bank' / target
    if target == 'extra.json':
        path.write_text('{}')
    else:
        value = ev.read_json(path)
        value['tampered'] = True
        put(path, value)
    with pytest.raises(ValueError):
        ev.validate_bank(plan)


def test_bank_requires_exact_source_population_and_dependency_versions(tmp_path):
    f = fixture(tmp_path)
    plan, receipt, records = bank_fixture(f)
    records[-1] = records[0]
    with pytest.raises(ValueError, match='identity'):
        ev.seal_bank(plan, receipt, records)
    assert not (f['case'] / 'evaluation/bank').exists()
    receipt['generator_dependency_versions'].pop('peft')
    with pytest.raises(ValueError, match='versions'):
        ev.seal_bank(plan, receipt, records)


def test_executor_infrastructure_error_after_warning_is_fatal():
    with pytest.raises(RuntimeError, match='namespace'):
        ev._infrastructure_check({'outcome': 'RUNTIME_EXCEPTION', 'stderr': 'warning\nbwrap: namespace unavailable'})


def test_probe_hashes_the_executors_fixed_path_binary(monkeypatch):
    from types import SimpleNamespace
    hashed = []
    monkeypatch.setattr(ev.shutil, 'which', lambda name, path=None: '/usr/bin/bwrap' if path == '/usr/bin:/bin' else '/shadow/bwrap')
    monkeypatch.setattr(ev, '_pbpf_module', lambda *args: SimpleNamespace(
        execute_stdin=lambda *a, **k: {'outcome': 'PASS'}, __file__='/pinned/rbr_execution.py'))
    def hash_path(path):
        hashed.append(str(path))
        return 'a' * 64
    monkeypatch.setattr(ev, 'file_sha', hash_path)
    ev.probe_sandbox('/pinned', 'rbr')
    assert '/usr/bin/bwrap' in hashed
    assert '/shadow/bwrap' not in hashed


def test_candidate_failure_reprobes_converted_infrastructure_error(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(ev, '_pbpf_module', lambda *args: SimpleNamespace(execute_call=lambda *a, **k: {'outcome': 'RUNTIME_EXCEPTION'}))
    def failed_probe(*args):
        raise RuntimeError('sandbox interpreter failed')
    monkeypatch.setattr(ev, 'probe_sandbox', failed_probe)
    private = {'tests': [{'id': str(i), 'input': 'print(1)', 'expected': '1', 'expected_error': False} for i in range(10)]}
    plan = dict(domain='codearc', pbpf_root='unused', timeout_seconds=6)
    with pytest.raises(RuntimeError, match='interpreter'):
        ev._evaluate_candidate((plan, private, {'program': 'pass'}))
