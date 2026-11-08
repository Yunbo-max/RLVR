"""Prospective matched evaluation of the twelve registered EESD rebuttal cells.

This is a NEW protocol, not historical exact replication. Only pinned PBPF task
semantics and isolated executors are reused. Preparation needs no GPU libraries;
model loading is local-only and occurs after a successful sandbox preflight.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

PBPF_PIN = '0200283849274b49e303638b91f142acd8bba984'
PLAN_SCHEMA = 'eesd-rebuttal-per-case-matched-evaluator-v1'
REPORT_SCHEMA = 'eesd-fresh-policy-eval-v1'
BANK_SCHEMA = 'eesd-rebuttal-matched-generation-v1'
EVALUATION_PROTOCOL = 'eesd-rebuttal-matched-prospective-v1'
ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = ROOT / 'scripts/run_eesd_rebuttal_matched_evaluation.py'
MODELS = {
    'qwen25_7b': ('Qwen/Qwen2.5-Coder-7B-Instruct', 'c03e6d358207e414f1eca0bb1891e29f1db0e242'),
    'deepseek_6p7b': ('deepseek-ai/deepseek-coder-6.7b-instruct', 'e5d64addd26a6a1db0f9b863abf6ee3141936807'),
    'gemma3_4b': ('google/gemma-3-4b-it', '093f9f388b31de276ce2de164bdc2081324b9767'),
}
DATASETS = {'runbugrun': 'rbr', 'codearc': 'codearc', 'apps_replay': 'apps_replay', 'codecontests_replay': 'codecontests_replay'}
PROTOCOL = dict(evaluation_protocol=EVALUATION_PROTOCOL, prospective_protocol=EVALUATION_PROTOCOL)
QUANTIZATION = dict(load_in_4bit=True, bnb_4bit_compute_dtype='bfloat16', bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
PBPF_FILES = (
    'src/pbpf/apbpf/rbr_prompt.py', 'src/pbpf/apbpf/codearc_prompt.py',
    'src/pbpf/apbpf/codearc_bank.py', 'src/pbpf/apbpf/rbr_execution.py',
    'src/pbpf/apbpf/codearc_execution.py', 'src/pbpf/apbpf/rbr_materialize.py',
    'src/pbpf/apbpf/codearc_materialize.py', 'src/pbpf/eesd/replay_prompt.py',
    'src/pbpf/eesd/replay_materialization.py', 'src/pbpf/eesd/replay_admission.py',
)


def now():
    return datetime.now(timezone.utc).isoformat()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tree_sha(directory):
    root = Path(directory)
    files = sorted(p for p in root.rglob('*') if p.is_file())
    if not files:
        raise ValueError('adapter tree must be nonempty')
    digest = hashlib.sha256()
    for path in files:
        if path.is_symlink():
            raise ValueError('adapter files must not be symlinks')
        name = path.relative_to(root).as_posix().encode()
        digest.update(len(name).to_bytes(4, 'big'))
        digest.update(name)
        digest.update(bytes.fromhex(file_sha(path)))
    return digest.hexdigest()


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f'duplicate JSON key: {key}')
        value[key] = item
    return value


def read_json(path):
    return json.loads(Path(path).read_bytes().decode('utf-8'), object_pairs_hook=_unique_object)


def read_jsonl(path):
    # splitlines() would corrupt literal U+0085/U+2028 inside JSON strings.
    rows = [json.loads(line, object_pairs_hook=_unique_object)
            for line in Path(path).read_bytes().decode('utf-8').split('\n') if line.strip()]
    if not rows or any(not isinstance(r, dict) for r in rows):
        raise ValueError(f'nonempty JSON object rows required: {path}')
    return rows


def write_json(path, value, *, exclusive=True):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
    else:
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text(text, encoding='utf-8', newline='\n')
        temporary.replace(path)


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _string(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'nonempty string required for {field}')
    return value


def _normalized_io(value):
    return '\n'.join(line.rstrip() for line in value.replace('\r\n', '\n').strip().splitlines())


def pair_sha(stdin, stdout):
    """Original Replay admission fingerprint; never normalize executed IO."""
    return hashlib.sha256((_normalized_io(stdin) + '\0' + _normalized_io(stdout)).encode()).hexdigest()


def validate_checkout(pbpf):
    pbpf = Path(pbpf).resolve()
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=pbpf, text=True, capture_output=True, check=True).stdout.strip()
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=pbpf, text=True, capture_output=True, check=True).stdout.strip()
    if head != PBPF_PIN or status:
        raise ValueError(f'PBPF must be clean and pinned at {PBPF_PIN}')
    return dict(pbpf_commit=head, pbpf_source_hashes={name: file_sha(pbpf / name) for name in PBPF_FILES})


def _validate_identity_manifest(manifest_path, *, receipt, audit, cell, model, revision, dataset):
    """Sealed per-cell identity manifest (the README's separate original-source
    manifest). Only fills the dataset_identity gap; model identity must already
    be verified by the receipt audit. The manifest binds the corrections sha
    and the cell/model/dataset triple; it is an auxiliary sealed artifact, not
    a substitute for per-row model identity."""
    manifest_path = Path(manifest_path).resolve()
    value = read_json(manifest_path)
    if value.get('schema') != 'eesd-iclr2027-cell-identity-manifest-v1':
        raise ValueError('identity manifest schema mismatch')
    corrections_sha = audit.get('source_sha256')
    if (value.get('corrections_sha256') != corrections_sha
            or value.get('cell') != cell or value.get('dataset') != dataset
            or value.get('model_id') != model or value.get('model_revision') != revision):
        raise ValueError('identity manifest does not bind the frozen corrections/cell/model')
    if not isinstance(value.get('prepare_receipt_sha256'), str) or len(value['prepare_receipt_sha256']) != 64:
        raise ValueError('identity manifest lacks the prepare receipt binding')
    return dict(path=str(manifest_path), sha256=file_sha(manifest_path))


def _case_identity(case, domain=None, identity_manifest=None):
    import yaml
    receipt = read_json(case / 'case.json')
    if receipt.get('schema') != 'eesd-iclr2027-independent-case-v1':
        raise ValueError('invalid independent case receipt')
    kind = receipt.get('case', {}).get('kind')
    expected_status = {'train': 'trained_awaiting_independent_evaluation', 'baseline': 'completed_diagnostic_without_gpu'}
    if kind not in expected_status or receipt.get('status') != expected_status[kind]:
        raise ValueError('case has no verified training/baseline adapter')
    cell = receipt.get('original_paper_cell')
    matrix = read_json(ROOT / 'configs/rebuttal/model_dataset_matrix.json')
    profiles = [c for c in matrix['cells'] if c['cell'] == cell]
    if len(profiles) != 1:
        raise ValueError('case must name exactly one registered original model/dataset cell')
    profile = profiles[0]
    if (profile['dataset'] not in DATASETS or profile['model'] not in MODELS
            or cell != profile['dataset'] + '__' + profile['model']
            or profile['domain'] != DATASETS[profile['dataset']]
            or (domain is not None and domain != profile['domain'])):
        raise ValueError('cell/domain mismatch')
    setup = receipt['setup']
    model_path = Path(setup['model_config_path']).resolve()
    if file_sha(model_path) != setup.get('model_config_sha256'):
        raise ValueError('model config changed since case preparation')
    cfg = yaml.safe_load(model_path.read_bytes().decode('utf-8'))
    model, revision = MODELS[profile['model']]
    if (cfg.get('model_id') != model or cfg.get('revision') != revision
            or setup.get('model_id') != model or setup.get('revision') != revision
            or profile['model_id'] != model
            or file_sha(ROOT / profile['model_config']) != file_sha(model_path)):
        raise ValueError('registered model/revision/config differs from prepared case')
    provenance_path = case / 'data_provenance.json'
    if file_sha(provenance_path) != receipt.get('data_provenance_sha256'):
        raise ValueError('case data provenance changed')
    provenance = read_json(provenance_path)
    if provenance.get('errors'):
        raise ValueError('case provenance contains input errors')
    audit = receipt.get('cell_identity_audit', {})
    manifest_binding = None
    if (audit.get('cell') != cell or audit.get('model_id') != model
            or audit.get('model_revision') != revision
            or audit.get('dataset') != profile['dataset']):
        raise ValueError('case model/dataset identity audit is incomplete or mismatched')
    if audit.get('status') != 'verified_model_and_dataset_identity':
        if (audit.get('status') == 'incomplete_record_identity_metadata'
                and audit.get('missing_fields') == {'dataset_identity': audit.get('rows')}
                and identity_manifest is not None):
            manifest_binding = _validate_identity_manifest(
                identity_manifest, receipt=receipt, audit=audit, cell=cell,
                model=model, revision=revision, dataset=profile['dataset'])
        else:
            raise ValueError('case model/dataset identity audit is incomplete or mismatched')
    adapter = (case / 'model/adapter') if kind == 'train' else Path(setup['shared_init_adapter_path'])
    adapter = adapter.resolve()
    expected = receipt.get('trained_adapter_tree_sha256') if kind == 'train' else setup.get('shared_init_adapter_tree_sha256')
    if tree_sha(adapter) != expected:
        raise ValueError('adapter differs from frozen case outcome')
    from eesd.training_outcomes import adapter_metadata
    adapter_metadata(adapter, receipt['setup']['model_config_path'])
    return receipt, profile, cfg, provenance, adapter, manifest_binding


def _validate_tests(public, private, domain):
    replay = domain.endswith('_replay')
    visible, tests = public.get('visible_tests'), private.get('tests')
    if not isinstance(visible, list) or len(visible) != 4 or not isinstance(tests, list) or len(tests) != 10:
        raise ValueError('original task must have exactly four visible and six private tests')
    if any(not isinstance(t, dict) for t in visible + tests) or [t.get('id') for t in tests] != [str(i) for i in range(10)]:
        raise ValueError('duplicate/missing/private test identity')
    output_key = 'output' if replay else 'expected'
    allowed_visible = {'input', 'output'} if replay else {'id', 'input', 'expected'} | ({'expected_error'} if domain == 'codearc' else set())
    if any(set(t) != allowed_visible for t in visible):
        raise ValueError('unexpected public test fields or hidden leakage')
    if not replay and [t['id'] for t in visible] != ['0', '1', '2', '3']:
        raise ValueError('duplicate/missing public test identity')
    observed_inputs = set()
    exposed = private.get('known_exposed_input_sha256', [])
    if replay and (not isinstance(exposed, list) or len(set(exposed)) != len(exposed)):
        raise ValueError('invalid Replay exposed-input fingerprint list')
    for index, test in enumerate(tests):
        if any(not isinstance(test.get(k), str) for k in ('input', output_key)):
            raise ValueError('test input/output must preserve original strings')
        key = _normalized_io(test['input']) if replay else test['input']
        if replay and key in observed_inputs:
            raise ValueError('duplicate test input')
        observed_inputs.add(key)
        if replay:
            pools = {'apps_input_output'} if domain == 'apps_replay' else {'public_tests', 'private_tests', 'generated_tests'}
            if (test.get('pair_sha256') != pair_sha(test['input'], test['output'])
                    or not isinstance(test.get('original_pools'), list) or not test['original_pools']
                    or not set(test['original_pools']) <= pools
                    or type(test.get('known_exposed_input')) is not bool):
                raise ValueError('Replay test fingerprint/provenance mismatch')
            known = hashlib.sha256(_normalized_io(test['input']).encode()).hexdigest() in exposed
            if test['known_exposed_input'] != known or (index >= 4 and known):
                raise ValueError('Replay private test was publicly exposed')
        elif type(test.get('hidden')) is not bool or test['hidden'] != (index >= 4):
            raise ValueError('hidden/visible test designation mismatch')
        if domain == 'codearc' and type(test.get('expected_error')) is not bool:
            raise ValueError('CodeARC expected_error must be boolean')
        if index < 4 and any(visible[index].get(k) != test.get(k) for k in allowed_visible):
            raise ValueError('public/private visible observations differ')


def inventory(public, evaluator, domain, expected_sources=500):
    """Validate both original manifests, all identities and the primary population."""
    public, evaluator = Path(public).resolve(), Path(evaluator).resolve()
    pm, em = read_json(public / 'manifest.json'), read_json(evaluator / 'manifest.json')
    ph, eh = file_sha(public / 'tasks.jsonl'), file_sha(evaluator / 'tasks.jsonl')
    extra = {}
    if domain.endswith('_replay'):
        for folder, manifest, schema, sha in [(public, pm, 'eesd-replay-public-v1', ph), (evaluator, em, 'eesd-replay-evaluator-v1', eh)]:
            if (manifest.get('schema') != schema or manifest.get('domain') != domain
                    or manifest.get('task_kind') != 'stdin_synthesis' or manifest.get('tasks', {}).get('sha256') != sha
                    or manifest.get('tasks', {}).get('bytes') != (folder / 'tasks.jsonl').stat().st_size):
                raise ValueError('Replay materialization manifest/hash/domain mismatch')
        if public.parent != evaluator.parent or public.parent.name != domain:
            raise ValueError('Replay public/evaluator roots must share their admitted domain directory')
        bundle = public.parent.parent
        admission = read_json(bundle / 'admission.json')
        if admission.get('schema') != 'eesd-replay-admission-v1' or admission.get('status') != 'data-admitted-not-generated-not-scored':
            raise ValueError('sealed original Replay admission required')
        for path in [public / 'manifest.json', public / 'tasks.jsonl', evaluator / 'manifest.json', evaluator / 'tasks.jsonl']:
            locked = admission.get('outputs', {}).get(path.relative_to(bundle).as_posix(), {})
            if locked.get('sha256') != file_sha(path) or locked.get('bytes') != path.stat().st_size:
                raise ValueError('Replay admission/public/private origin binding mismatch')
        extra = dict(admission_path=str(bundle / 'admission.json'), admission_sha256=file_sha(bundle / 'admission.json'))
    else:
        schema = {'rbr': 'apbpf-rbr-generated-materialization-v1', 'codearc': 'apbpf-codearc-replay-materialization-v1'}.get(domain)
        if (schema is None or pm.get('schema') != schema or em.get('schema') != schema
                or pm.get('public_tasks_sha256') != ph or em.get('public_tasks_sha256') != ph
                or em.get('evaluator_tasks_sha256') != eh):
            raise ValueError('original public/private manifest schema or task hash mismatch')
    pubs, privs = read_jsonl(public / 'tasks.jsonl'), read_jsonl(evaluator / 'tasks.jsonl')
    private_by_task = {}
    for row in privs:
        task = _string(row.get('task_id'), 'private task_id')
        if task in private_by_task:
            raise ValueError('duplicate private task identity')
        private_by_task[task] = row
    seen, source_splits, primary = set(), {}, {}
    replay = domain.endswith('_replay')
    identity_keys = {'task_id', 'source_component_id', 'split', 'domain' if replay else 'protocol'} | ({'base_bug_id'} if domain == 'rbr' else set())
    public_fields = identity_keys | ({'statement', 'visible_tests'} if replay else {'task_text', 'visible_tests'}) | ({'buggy_code'} if domain == 'rbr' else set())
    for row in pubs:
        if set(row) != public_fields:
            raise ValueError('unexpected public task fields or hidden leakage')
        task = _string(row['task_id'], 'public task_id')
        source = _string(row['source_component_id'], 'source_component_id')
        split = row['split']
        if split not in {'train', 'development', 'primary'} or task in seen:
            raise ValueError('duplicate task identity or invalid split')
        seen.add(task)
        private = private_by_task.get(task)
        if private is None or any(private.get(k) != row[k] for k in identity_keys):
            raise ValueError('public/private task identity mismatch')
        if replay:
            if row['domain'] != domain or private.get('statement') != row['statement']:
                raise ValueError('Replay domain/statement identity mismatch')
        elif row['protocol'] != ('RBR-generated-repair' if domain == 'rbr' else 'CodeARC-Replay'):
            raise ValueError('public task domain/protocol mismatch')
        _string(row.get('statement' if replay else 'task_text'), 'public task text')
        _validate_tests(row, private, domain)
        if source in source_splits and (source_splits[source] != split or domain != 'codearc'):
            raise ValueError('duplicate source or source split overlap')
        source_splits[source] = split
        if split == 'primary' and (source not in primary or task < primary[source]['task_id']):
            primary[source] = row
    if seen != set(private_by_task):
        raise ValueError('public/private task populations differ')
    if len(primary) != expected_sources:
        raise ValueError(f'original primary source population must contain {expected_sources} sources')
    if replay:
        counts = {s: sum(r['split'] == s for r in pubs) for s in set(r['split'] for r in pubs)}
        if pm.get('counts') != counts or em.get('counts') != counts:
            raise ValueError('Replay manifest split counts differ')
    selected = [primary[source] for source in sorted(primary)]
    return dict(public_manifest_sha256=file_sha(public / 'manifest.json'), public_tasks_sha256=ph,
                evaluator_manifest_sha256=file_sha(evaluator / 'manifest.json'), evaluator_tasks_sha256=eh,
                evaluation_source_ids=sorted(primary), evaluation_sources=len(primary),
                primary_tasks=[{k: r[k] for k in ('task_id', 'source_component_id')} for r in selected], **extra)


def _dataset_fields_match(row, dataset):
    # Same aliases as the independent case's strict correction identity audit.
    aliases = {key: key for key in DATASETS}
    aliases['rbr'] = 'runbugrun'
    return all(isinstance(row[key], str) and aliases.get(row[key]) == dataset
               for key in ('dataset', 'source_dataset', 'domain') if key in row)


def _mapping_and_disjointness(case, receipt, profile, provenance, eval_ids, inv):
    expected = receipt.get('frozen_case_input_hashes', {}).get('eval_ids_sha256')
    if eval_ids is None or not expected or file_sha(eval_ids) != expected:
        raise ValueError('evaluation source-ID file must match original prepared CASE hash')
    rows = read_jsonl(eval_ids)
    sources = [_string(r.get('source_component_id'), 'evaluation source-ID') for r in rows]
    if len(set(sources)) != len(sources) or sorted(sources) != inv['evaluation_source_ids']:
        raise ValueError('duplicate or mismatched evaluation source-ID population')
    selected = {r['source_component_id']: r['task_id'] for r in inv['primary_tasks']}
    for row in rows:
        if not _dataset_fields_match(row, profile['dataset']):
            raise ValueError('evaluation source-ID dataset/domain mismatch')
        for key, expected_value in [('cell', profile['cell']), ('model_id', profile['model_id'])]:
            if key in row and row[key] != expected_value:
                raise ValueError(f'evaluation source-ID {key} mismatch')
        if 'task_id' in row and row['task_id'] != selected[row['source_component_id']]:
            raise ValueError('evaluation source-ID task mapping differs from selected primary task')
        if 'problem_id' in row and (type(row['problem_id']) not in (str, int) or not str(row['problem_id']).strip()):
            raise ValueError('invalid supplied problem ID; never derive missing problem IDs')
    pe = provenance.get('evaluation', {})
    if pe.get('eval_input_sha256') != expected or pe.get('source_level_overlap_checked') is not True:
        raise ValueError('case provenance does not bind the evaluation source-ID file')
    setup = receipt['setup']
    corrections_path = setup.get('corrections_path')
    if not corrections_path:
        raise ValueError('case must freeze corrections_path and corrections_sha256 to recheck disjointness; reprepare baseline receipt')
    corrections = Path(corrections_path).resolve()
    ch = file_sha(corrections)
    if ch != setup.get('corrections_sha256') or ch != provenance.get('correction_file_sha256') or ch != receipt['cell_identity_audit'].get('source_sha256'):
        raise ValueError('correction input identity/hash differs from original prepared case')
    train = read_jsonl(corrections)
    train_sources, train_problems = {}, {}
    all_problems = True
    for row in train:
        source, split = _string(row.get('source_component_id'), 'training source ID'), row.get('split')
        if split not in {'train', 'development'}:
            raise ValueError('corrections must contain train/development only')
        if source in train_sources and train_sources[source] != split:
            raise ValueError('training/development source overlap')
        train_sources[source] = split
        if row.get('model_id') != profile['model_id'] or row.get('model_revision') != MODELS[profile['model']][1]:
            raise ValueError('correction model/revision differs from case')
        if not _dataset_fields_match(row, profile['dataset']):
            raise ValueError('correction dataset/domain differs from case')
        problem = row.get('problem_id')
        if problem is None:
            all_problems = False
        else:
            if type(problem) not in (str, int) or not str(problem).strip():
                raise ValueError('invalid training problem ID')
            problem = str(problem)
            if problem in train_problems and train_problems[problem] != split:
                raise ValueError('training/development problem overlap')
            train_problems[problem] = split
    if set(sources) & set(train_sources):
        raise ValueError('evaluation/training/development source overlap')
    eval_problems = {str(r['problem_id']) for r in rows if r.get('problem_id') is not None}
    if eval_problems & set(train_problems):
        raise ValueError('evaluation/training/development problem overlap')
    verified_problems = all_problems and all(r.get('problem_id') is not None for r in rows)
    if pe.get('problem_level_overlap_checked') and not verified_problems:
        raise ValueError('case claims problem disjointness without original problem IDs')
    return dict(eval_source_id_file_path=str(Path(eval_ids).resolve()), eval_source_id_file_sha256=expected,
                corrections_path=str(corrections), corrections_sha256=ch,
                source_disjointness_status=provenance.get('provenance_status', 'not_certified'),
                source_disjointness_rechecked=True, problem_disjointness_rechecked=verified_problems,
                problem_id_policy='only supplied original problem IDs; absent IDs remain unverified')


def generation_policy(domain):
    return dict(candidates=1, split='primary', decode_policy='greedy', do_sample=False, num_beams=1,
                max_input_tokens=4096, max_new_tokens=512 if domain == 'codearc' else 1024,
                quantization=QUANTIZATION, torch_dtype='bfloat16', local_files_only=True,
                prompt_policy='pinned PBPF public task prompts; Gemma system content folded into user',
                input_tokenization='rendered chat template; add_special_tokens=False; no truncation',
                primary_task_policy='lexicographically first task_id per source; sorted source order',
                task_kind='replay_synthesis' if domain.endswith('_replay') else 'repair')


def _code_hashes():
    paths = [Path(__file__).resolve(), ENTRYPOINT,
             ROOT / 'scripts/run_eesd_weighted_sft.py', ROOT / 'src/eesd/training_outcomes.py',
             ROOT / 'configs/rebuttal/model_dataset_matrix.json']
    return {str(p): file_sha(p) for p in paths}


def prepare(case, pbpf, public, evaluator, workers=1, eval_ids=None, domain=None, identity_manifest=None):
    case, pbpf, public, evaluator = [Path(p).resolve() for p in (case, pbpf, public, evaluator)]
    if type(workers) is not int or not 1 <= workers <= 12:
        raise ValueError('workers must be an integer from 1 through 12')
    output = case / 'evaluation'
    if output.exists():
        raise FileExistsError('immutable case evaluator already prepared')
    receipt, profile, cfg, provenance, adapter, manifest_binding = _case_identity(case, domain, identity_manifest)
    pin = validate_checkout(pbpf)
    inv = inventory(public, evaluator, profile['domain'], profile['original_paper']['N'])
    mapping = _mapping_and_disjointness(case, receipt, profile, provenance, eval_ids, inv)
    plan = dict(schema=PLAN_SCHEMA, status='prepared', **PROTOCOL,
                case_dir=str(case), case_sha256=file_sha(case / 'case.json'), case_id=receipt['case']['id'],
                study=receipt['study'], seed=receipt['seed'], original_paper_cell=profile['cell'],
                model_id=cfg['model_id'], revision=cfg['revision'], family=profile['model'],
                domain=profile['domain'], dataset=profile['dataset'], task_kind=profile['kind'],
                adapter_path=str(adapter), adapter_tree_sha256=tree_sha(adapter),
                model_config_path=str(Path(receipt['setup']['model_config_path']).resolve()),
                model_config_sha256=receipt['setup']['model_config_sha256'],
                data_provenance_sha256=file_sha(case / 'data_provenance.json'),
                pbpf_root=str(pbpf), public_root=str(public), evaluator_root=str(evaluator),
                workers=workers, timeout_seconds=6.0, execution_profile='sandbox',
                generation_policy=generation_policy(profile['domain']),
                generated_program_count_policy='one fresh greedy candidate per independent PRIMARY source',
                evaluation_policy='exact original four visible plus six private tests; isolated execution only',
                historical_replication=False,
                protocol_differences=['common NF4/BF16 model loading', 'common greedy decoding and input tokenization',
                                      'Gemma system instructions folded into user', 'Replay uses isolated PBPF stdin executor'],
                identity_manifest=manifest_binding,
                local_source_hashes=_code_hashes(), **pin, **inv, **mapping,
                scientific_status='prospective protocol prepared; generation/evaluation NOT YET RUN', created_at=now())
    plan['generation_command'] = [sys.executable, str(ENTRYPOINT), '--case-dir', str(case), '--_phase', 'generate']
    plan['heldout_command'] = [sys.executable, str(ENTRYPOINT), '--case-dir', str(case), '--_phase', 'evaluate']
    plan['frozen_plan_sha256'] = _json_sha(_frozen(plan))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', plan)
    return plan


MUTABLE_PLAN_FIELDS = {'status', 'scientific_status', 'started_at', 'finished_at', 'failed_at', 'error',
                       'heldout_report_sha256', 'heldout_all_tests_pass_at_1', 'heldout_sources', 'heldout_count',
                       'runtime_sha256', 'bank_complete_sha256', 'frozen_plan_sha256'}


def _frozen(plan):
    return {k: v for k, v in plan.items() if k not in MUTABLE_PLAN_FIELDS}


def check_frozen(plan, case):
    case = Path(case).resolve()
    if (plan.get('schema') != PLAN_SCHEMA or any(plan.get(k) != v for k, v in PROTOCOL.items())
            or _json_sha(_frozen(plan)) != plan.get('frozen_plan_sha256') or plan.get('case_dir') != str(case)):
        raise ValueError('prepared prospective plan identity/content changed')
    if file_sha(case / 'case.json') != plan['case_sha256']:
        raise ValueError('case receipt changed after evaluator preparation')
    receipt, profile, cfg, provenance, adapter, _manifest = _case_identity(case, plan['domain'])
    for key, value in dict(original_paper_cell=profile['cell'], model_id=cfg['model_id'], revision=cfg['revision'],
                           adapter_path=str(adapter), adapter_tree_sha256=tree_sha(adapter),
                           data_provenance_sha256=file_sha(case / 'data_provenance.json'),
                           generation_policy=generation_policy(plan['domain'])).items():
        if plan.get(key) != value:
            raise ValueError(f'prepared {key} changed')
    if _code_hashes() != plan['local_source_hashes']:
        raise ValueError('evaluator/model-loader/registry source changed after preparation')
    observed = {**validate_checkout(plan['pbpf_root']), **inventory(plan['public_root'], plan['evaluator_root'], plan['domain'], profile['original_paper']['N'])}
    observed.update(_mapping_and_disjointness(case, receipt, profile, provenance, Path(plan['eval_source_id_file_path']), observed))
    if any(plan.get(k) != v for k, v in observed.items()):
        raise ValueError('frozen evaluator inputs or source-ID map changed')


def _pbpf_module(root, name):
    """Import only the frozen checkout, without modifying its read-only tree."""
    source = (Path(root) / 'src').resolve()
    for key, module in list(sys.modules.items()):
        if key == 'pbpf' or key.startswith('pbpf.'):
            origin = getattr(module, '__file__', None)
            if origin and not Path(origin).resolve().is_relative_to(source):
                raise RuntimeError('another PBPF installation is already imported')
    sys.dont_write_bytecode = True
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    module = importlib.import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(source):
        raise RuntimeError('PBPF module did not resolve inside the pinned checkout')
    return module


class _PublicTokenizer:
    """Common tokenization and the explicit Gemma role adaptation."""

    def __init__(self, tokenizer, family):
        self.tokenizer, self.family = tokenizer, family

    def encode(self, text, **kwargs):
        kwargs['add_special_tokens'] = False
        return self.tokenizer.encode(text, **kwargs)

    def apply_chat_template(self, messages, **kwargs):
        messages = [dict(message) for message in messages]
        if self.family == 'gemma3_4b' and messages and messages[0]['role'] == 'system':
            if len(messages) < 2 or messages[1]['role'] != 'user':
                raise ValueError('Gemma public prompt must have a user message after system instructions')
            messages[1]['content'] = messages[0]['content'] + '\n\n' + messages[1]['content']
            messages = messages[1:]
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


def public_prompt(tokenizer, row, plan):
    """Render only a whitelisted original public record, never private tests."""
    domain, family = plan['domain'], plan['family']
    common = {'task_id', 'source_component_id', 'split'}
    fields = (common | {'domain', 'statement', 'visible_tests'} if domain.endswith('_replay')
              else common | {'protocol', 'task_text', 'visible_tests'}
              | ({'base_bug_id', 'buggy_code'} if domain == 'rbr' else set()))
    if set(row) != fields or row.get('split') != 'primary':
        raise ValueError('generation accepts only whitelisted primary public task records')
    visible = row['visible_tests']
    keys = ({'input', 'output'} if domain.endswith('_replay') else
            {'id', 'input', 'expected'} | ({'expected_error'} if domain == 'codearc' else set()))
    if (not isinstance(visible, list) or len(visible) != 4
            or any(not isinstance(test, dict) or set(test) != keys for test in visible)):
        raise ValueError('exactly four whitelisted public examples required')
    adapted = _PublicTokenizer(tokenizer, family)
    maximum = plan['generation_policy']['max_input_tokens']
    if domain.endswith('_replay'):
        if row['domain'] != domain:
            raise ValueError('Replay public prompt domain mismatch')
        module = _pbpf_module(plan['pbpf_root'], 'pbpf.eesd.replay_prompt')
        messages = module.messages_for(row['statement'], visible)
        prompt = adapted.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audit = dict(field_character_cap=None, clipped_fields=[], public_example_ids=['0', '1', '2', '3'],
                     replay_prompt_policy=module.POLICY)
    else:
        module = _pbpf_module(plan['pbpf_root'], f'pbpf.apbpf.{domain}_prompt')
        prompt, audit = module.prompt_for(adapted, row, family=family, max_input_tokens=maximum)
    ids = list(adapted.encode(prompt))
    if not ids or len(ids) > maximum:
        raise ValueError('public prompt exceeds the common input cap; Replay text is never silently truncated')
    if any(type(token) is not int or token < 0 for token in ids):
        raise ValueError('tokenizer returned invalid token IDs')
    audit = dict(audit, input_tokens=len(ids), prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                 input_ids_sha256=_json_sha(ids), add_special_tokens=False,
                 system_role_policy='fold_into_user' if family == 'gemma3_4b' else 'native')
    return prompt, ids, audit


def _infrastructure_check(result):
    if not isinstance(result, dict) or result.get('outcome') not in {
        'PASS', 'WRONG_OUTPUT', 'COMPILE_ERROR', 'RUNTIME_EXCEPTION', 'TIMEOUT'
    }:
        raise RuntimeError('isolated executor returned an invalid result')
    # The pinned executors reject a leading bwrap diagnostic. Also catch it
    # after startup warnings; never report a namespace failure as model failure.
    for line in str(result.get('stderr', '')).splitlines():
        if line.lstrip().startswith('bwrap:'):
            raise RuntimeError(f'sandbox infrastructure failure: {line}')


def probe_sandbox(pbpf, domain, timeout=6.0):
    """Harmless isolated startup probe. No direct-execution fallback exists."""
    # Both pinned executors supply this fixed PATH to subprocess.Popen.
    executable = shutil.which('bwrap', path='/usr/bin:/bin')
    if not executable or not Path('/usr/bin/python3').is_file():
        raise RuntimeError('isolated evaluation requires bwrap and /usr/bin/python3')
    if domain == 'codearc':
        module = _pbpf_module(pbpf, 'pbpf.apbpf.codearc_execution')
        actual = module.execute_call('def solution(x):\n    return x\n',
                                     {'input': 'print(solution(17))', 'expected': '17', 'expected_error': False},
                                     timeout=timeout)
    elif domain in DATASETS.values():
        module = _pbpf_module(pbpf, 'pbpf.apbpf.rbr_execution')
        actual = module.execute_stdin('print(input())\n', {'input': '17\n', 'expected': '17\n'}, timeout=timeout)
    else:
        raise ValueError('unknown evaluation domain')
    _infrastructure_check(actual)
    if actual['outcome'] != 'PASS':
        raise RuntimeError(f'isolated startup probe failed: {actual}')
    policy = dict(profile='sandbox', timeout_seconds=timeout, unshare_all=True, network=False,
                  cpu_seconds=int(timeout + 0.999999), address_space_bytes=1024 ** 3,
                  file_size_bytes=8 * 1024 ** 2, open_files=64, process_group_kill=True,
                  stdin_policy='pinned canonical_stdin: append final newline only when absent',
                  comparison=('pinned CodeARC score_result' if domain == 'codearc' else
                              'pinned output_matches: text/token comparison, numeric absolute tolerance 1e-4'))
    identity = dict(execution_policy=policy,
                    bwrap_sha256=file_sha(Path(executable).resolve()),
                    python_sha256=file_sha(Path('/usr/bin/python3').resolve()),
                    executor_sha256=file_sha(module.__file__),
                    shared_limits_sha256=file_sha(Path(pbpf) / 'src/pbpf/apbpf/codearc_execution.py'),
                    kernel=os.uname().release)
    return dict(**identity, execution_lock_sha256=_json_sha(identity), probe_outcome='PASS')


def run_stage(command, log_path):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                       PYTHONPATH=str(ROOT / 'src') + os.pathsep + os.environ.get('PYTHONPATH', ''))
    with Path(log_path).open('x', encoding='utf-8') as log:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)


def _runtime(plan):
    path = Path(plan['case_dir']) / 'evaluation/runtime.json'
    if file_sha(path) != plan.get('runtime_sha256'):
        raise ValueError('sandbox runtime receipt changed')
    value = read_json(path)
    if (value.get('frozen_plan_sha256') != plan['frozen_plan_sha256']
            or value.get('evaluation_protocol') != EVALUATION_PROTOCOL
            or value.get('probe_outcome') != 'PASS'):
        raise ValueError('successful sandbox preflight receipt required')
    identity = {key: value[key] for key in ('execution_policy', 'bwrap_sha256', 'python_sha256',
                                          'executor_sha256', 'shared_limits_sha256', 'kernel')}
    if _json_sha(identity) != value.get('execution_lock_sha256'):
        raise ValueError('sandbox execution identity changed')
    return value


def _phase_plan(case, *, public_only=False):
    case = Path(case).resolve()
    plan = read_json(case / 'evaluation/plan.json')
    if plan.get('status') != 'running':
        raise ValueError('internal phase requires a running prepared evaluator')
    if public_only:
        # Generation deliberately never opens evaluator_root or its manifest.
        if (plan.get('schema') != PLAN_SCHEMA or plan.get('case_dir') != str(case)
                or any(plan.get(k) != v for k, v in PROTOCOL.items())
                or _json_sha(_frozen(plan)) != plan.get('frozen_plan_sha256')
                or file_sha(case / 'case.json') != plan['case_sha256']
                or _code_hashes() != plan['local_source_hashes']
                or tree_sha(plan['adapter_path']) != plan['adapter_tree_sha256']
                or file_sha(plan['model_config_path']) != plan['model_config_sha256']):
            raise ValueError('generation frozen identity changed')
        if any(plan.get(k) != v for k, v in validate_checkout(plan['pbpf_root']).items()):
            raise ValueError('generation PBPF source changed')
        for name, key in [('manifest.json', 'public_manifest_sha256'), ('tasks.jsonl', 'public_tasks_sha256')]:
            if file_sha(Path(plan['public_root']) / name) != plan[key]:
                raise ValueError('generation public input changed')
    else:
        check_frozen(plan, case)
    return plan, _runtime(plan)


def _bank_identity(plan):
    return dict(schema=BANK_SCHEMA, **PROTOCOL, frozen_plan_sha256=plan['frozen_plan_sha256'],
                model=plan['model_id'], revision=plan['revision'], domain=plan['domain'],
                task_kind=plan['task_kind'], adapter_sha256=plan['adapter_tree_sha256'],
                public_tasks_sha256=plan['public_tasks_sha256'],
                public_manifest_sha256=plan['public_manifest_sha256'],
                split='primary', candidates=1, decode_policy='greedy',
                generation_policy=plan['generation_policy'], sources=plan['evaluation_sources'])


def _candidate_contract(record, task, policy):
    if any(record.get(key) != task[key] for key in ('task_id', 'source_component_id')):
        raise ValueError('generated candidate identity differs from frozen primary task')
    if any(not isinstance(record.get(key), str) for key in ('completion', 'program')):
        raise ValueError('generated completion and extracted program must be strings')
    for key, lower, upper in [('input_tokens', 1, policy['max_input_tokens']),
                              ('output_tokens', 1, policy['max_new_tokens'])]:
        if type(record.get(key)) is not int or not lower <= record[key] <= upper:
            raise ValueError(f'invalid candidate {key}')
    digest = record.get('prompt_sha256')
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('candidate must bind the public prompt hash')
    if type(record.get('cutoff')) is not bool:
        raise ValueError('candidate must record whether generation hit its output cap')


def seal_bank(plan, run_receipt, candidate_records):
    """Write the immutable generation bank; also used in CPU contract tests."""
    records = list(candidate_records)
    if len(records) != len(plan['primary_tasks']):
        raise ValueError('one fresh candidate for every frozen primary source is required')
    versions = run_receipt.get('generator_dependency_versions')
    if not isinstance(versions, dict) or any(not isinstance(versions.get(k), str) or not versions[k]
                                             for k in ('torch', 'transformers', 'peft', 'bitsandbytes')):
        raise ValueError('actual generation dependency versions are required')
    if not isinstance(run_receipt.get('generation_runtime'), dict) or not run_receipt['generation_runtime']:
        raise ValueError('actual model loading and decoding runtime receipt is required')
    identity = _bank_identity(plan)
    if any(key in run_receipt and run_receipt[key] != value for key, value in identity.items()):
        raise ValueError('generation receipt differs from frozen model/protocol identity')
    for record, task in zip(records, plan['primary_tasks']):
        _candidate_contract(record, task, plan['generation_policy'])
    output = Path(plan['case_dir']) / 'evaluation/bank'
    output.mkdir(exist_ok=False)
    write_json(output / 'run.json', dict(run_receipt, **identity))
    files = {'run.json': file_sha(output / 'run.json')}
    for index, record in enumerate(records):
        name = f'{index:06d}.json'
        write_json(output / name, dict(record, schema='eesd-rebuttal-matched-candidate-v1', **PROTOCOL))
        files[name] = file_sha(output / name)
    complete = dict(schema='eesd-rebuttal-matched-generation-complete-v1', **PROTOCOL,
                    frozen_plan_sha256=plan['frozen_plan_sha256'], sources=len(records), files=files,
                    files_sha256=_json_sha(files), completed_at=now())
    write_json(output / 'complete.json', complete)
    return complete


def validate_bank(plan):
    output = Path(plan['case_dir']) / 'evaluation/bank'
    if plan.get('bank_complete_sha256') and file_sha(output / 'complete.json') != plan['bank_complete_sha256']:
        raise ValueError('generation bank seal changed after the generation phase')
    complete = read_json(output / 'complete.json')
    names = {'run.json'} | {f'{index:06d}.json' for index in range(plan['evaluation_sources'])}
    files = complete.get('files')
    if (complete.get('schema') != 'eesd-rebuttal-matched-generation-complete-v1'
            or set(complete) != {'schema', 'evaluation_protocol', 'prospective_protocol', 'frozen_plan_sha256',
                                 'sources', 'files', 'files_sha256', 'completed_at'}
            or any(complete.get(k) != v for k, v in PROTOCOL.items())
            or complete.get('frozen_plan_sha256') != plan['frozen_plan_sha256']
            or complete.get('sources') != plan['evaluation_sources']
            or not isinstance(files, dict) or set(files) != names
            or complete.get('files_sha256') != _json_sha(files)
            or {path.name for path in output.iterdir()} != names | {'complete.json'}):
        raise ValueError('generation bank is incomplete or has extra/mismatched records')
    if any((output / name).is_symlink() or file_sha(output / name) != value for name, value in files.items()):
        raise ValueError('generation bank content changed after sealing')
    run_receipt = read_json(output / 'run.json')
    if any(run_receipt.get(k) != v for k, v in _bank_identity(plan).items()):
        raise ValueError('generation bank belongs to a different adapter, cell or protocol')
    versions = run_receipt.get('generator_dependency_versions')
    if (not isinstance(versions, dict) or any(not isinstance(versions.get(k), str) or not versions[k]
                                            for k in ('torch', 'transformers', 'peft', 'bitsandbytes'))
            or not isinstance(run_receipt.get('generation_runtime'), dict) or not run_receipt['generation_runtime']):
        raise ValueError('generation bank has no runtime dependency receipt')
    records = []
    for index, task in enumerate(plan['primary_tasks']):
        record = read_json(output / f'{index:06d}.json')
        if record.get('schema') != 'eesd-rebuttal-matched-candidate-v1' or any(record.get(k) != v for k, v in PROTOCOL.items()):
            raise ValueError('generation candidate schema/protocol differs')
        _candidate_contract(record, task, plan['generation_policy'])
        records.append(record)
    return run_receipt, records, complete


def _verify_loaded_adapter(model, directory, torch):
    """Catch silently skipped/remapped PEFT tensors, especially Gemma versions."""
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    expected = load_file(str(Path(directory) / 'adapter_model.safetensors'), device='cpu')
    actual = get_peft_model_state_dict(model)
    if set(expected) != set(actual):
        raise ValueError('loaded adapter tensor keys differ; incompatible PEFT/transformers model layout')
    for key, tensor in expected.items():
        observed = actual[key].detach().cpu()
        if (tuple(tensor.shape) != tuple(observed.shape) or not torch.isfinite(tensor).all().item()
                or not torch.isfinite(observed).all().item()
                or not torch.equal(observed, tensor.to(dtype=observed.dtype))):
            raise ValueError(f'loaded adapter tensor does not match saved checkpoint: {key}')
    return dict(verification='all saved adapter tensor keys, shapes and values matched after loading',
                tensors=len(expected), autocast_adapter_dtype=False)


def generate(case):
    """Fresh local GPU inference. Heavy imports are confined to this phase."""
    import importlib.metadata
    import platform
    plan, runtime = _phase_plan(case, public_only=True)
    # Probe again in the generation process before importing/loading the ML stack.
    observed = probe_sandbox(plan['pbpf_root'], plan['domain'], plan['timeout_seconds'])
    if observed['execution_lock_sha256'] != runtime['execution_lock_sha256']:
        raise RuntimeError('sandbox environment changed before generation')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
    from peft import PeftModel
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('matched evaluation requires a CUDA GPU with BF16 support; CPU inference is not this protocol')
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft', 'bitsandbytes')}
    versions['python'] = platform.python_version()
    torch.manual_seed(plan['seed'])
    torch.cuda.manual_seed_all(plan['seed'])
    tokenizer = AutoTokenizer.from_pretrained(plan['model_id'], revision=plan['revision'], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError('pinned tokenizer has no pad/EOS token')
    tokenizer.padding_side = 'right'
    public_rows = {row['task_id']: row for row in read_jsonl(Path(plan['public_root']) / 'tasks.jsonl')}
    prompts = []
    for task in plan['primary_tasks']:
        row = public_rows[task['task_id']]
        if row['source_component_id'] != task['source_component_id']:
            raise ValueError('public source identity differs from frozen task selection')
        _, ids, audit = public_prompt(tokenizer, row, plan)
        prompts.append((task, ids, audit))
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                     bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
    model_class = AutoModelForCausalLM
    if plan['family'] == 'gemma3_4b':
        from transformers import Gemma3ForConditionalGeneration
        model_class = Gemma3ForConditionalGeneration
    base = model_class.from_pretrained(plan['model_id'], revision=plan['revision'], local_files_only=True,
                                        quantization_config=quantization, device_map={'': 0}, torch_dtype=torch.bfloat16)
    adapter_config = read_json(Path(plan['adapter_path']) / 'adapter_config.json')
    if plan['family'] == 'gemma3_4b':
        available = {name for name, _ in base.named_modules()}
        if any(target not in available for target in adapter_config['target_modules']):
            raise ValueError('Gemma adapter targets do not match this transformers layout; use the training environment')
    model = PeftModel.from_pretrained(base, plan['adapter_path'], is_trainable=False,
                                     local_files_only=True, autocast_adapter_dtype=False)
    adapter_audit = _verify_loaded_adapter(model, plan['adapter_path'], torch)
    model.eval()
    eos = getattr(base.generation_config, 'eos_token_id', None)
    if eos is None:
        eos = tokenizer.eos_token_id
    stops = [eos] if isinstance(eos, int) else eos
    if not stops or any(type(token) is not int or token < 0 for token in stops):
        raise ValueError('pinned model has no valid generation stop token IDs')
    generation_config = GenerationConfig(do_sample=False, num_beams=1, num_return_sequences=1,
                                          max_new_tokens=plan['generation_policy']['max_new_tokens'], min_new_tokens=0,
                                          pad_token_id=tokenizer.pad_token_id, eos_token_id=eos,
                                          bos_token_id=tokenizer.bos_token_id, use_cache=True,
                                          repetition_penalty=1.0, length_penalty=1.0)
    generation_runtime = dict(dependency_versions=versions, model_class=model_class.__name__,
                              cuda_version=torch.version.cuda, gpu_name=torch.cuda.get_device_name(0),
                              gpu_capability=list(torch.cuda.get_device_capability(0)),
                              quantization=QUANTIZATION, torch_dtype='bfloat16', device_map={'': 0},
                              local_files_only=True, do_sample=False, num_beams=1, num_return_sequences=1,
                              eos_token_ids=stops, pad_token_id=tokenizer.pad_token_id,
                              input_add_special_tokens=False, max_input_tokens=4096,
                              max_new_tokens=plan['generation_policy']['max_new_tokens'],
                              generation_config=generation_config.to_dict(), adapter_load=adapter_audit)
    extractor = _pbpf_module(plan['pbpf_root'], 'pbpf.apbpf.rbr_prompt').extract_program
    started, records = now(), []
    with torch.inference_mode():
        for index, (task, ids, audit) in enumerate(prompts):
            inputs = torch.tensor([ids], dtype=torch.long, device=model.device)
            output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                    generation_config=generation_config)
            if output.ndim != 2 or output.shape[0] != 1 or not torch.equal(output[0, :len(ids)], inputs[0]):
                raise RuntimeError('generation output does not preserve the input prefix')
            continuation = output[0, len(ids):].detach().cpu().tolist()
            if not continuation or len(continuation) > plan['generation_policy']['max_new_tokens']:
                raise RuntimeError('generation did not produce one bounded fresh continuation')
            completion = tokenizer.decode(continuation, skip_special_tokens=True)
            record = dict(task, completion=completion, program=extractor(completion),
                          input_tokens=len(ids), output_tokens=len(continuation), prompt_sha256=audit['prompt_sha256'],
                          prompt_metadata=audit, output_ids_sha256=_json_sha(continuation),
                          cutoff=continuation[-1] not in stops)
            records.append(record)
            print(json.dumps(dict(completed=index + 1, sources=len(prompts), task_id=task['task_id'],
                                  input_tokens=len(ids), output_tokens=len(continuation))), flush=True)
    # This public-only recheck never reads hidden tests into the model process.
    _phase_plan(case, public_only=True)
    seal_bank(plan, dict(generator_dependency_versions=versions, generation_runtime=generation_runtime,
                         started_at=started, finished_at=now()), records)


def _evaluate_candidate(payload):
    """Worker: only pinned isolated execution; exceptions remain fatal."""
    plan, private, candidate = payload
    execute = (_pbpf_module(plan['pbpf_root'], 'pbpf.apbpf.codearc_execution').execute_call
               if plan['domain'] == 'codearc' else
               _pbpf_module(plan['pbpf_root'], 'pbpf.apbpf.rbr_execution').execute_stdin)
    results = []
    for index, test in enumerate(private['tests']):
        executor_test = dict(test)
        if plan['domain'].endswith('_replay'):
            executor_test['expected'] = test['output']
        result = execute(candidate['program'], executor_test, timeout=plan['timeout_seconds'])
        _infrastructure_check(result)
        results.append(dict(result, test_id=test['id'], hidden=index >= 4))
    if any(result['outcome'] != 'PASS' for result in results):
        # CodeARC may convert a broken interpreter/wrapper into a candidate
        # failure. A known-good probe prevents such failures becoming scores.
        probe_sandbox(plan['pbpf_root'], plan['domain'], plan['timeout_seconds'])
    passes = [result['outcome'] == 'PASS' for result in results]
    return dict(task_id=candidate['task_id'], source_component_id=candidate['source_component_id'],
                all_tests_pass=all(passes), visible_all_pass=all(passes[:4]), hidden_all_pass=all(passes[4:]),
                test_pass_fraction=sum(passes) / len(passes), tests=results,
                input_tokens=candidate['input_tokens'], output_tokens=candidate['output_tokens'],
                cutoff=candidate['cutoff'], program_sha256=hashlib.sha256(candidate['program'].encode()).hexdigest())


def _records_contract(plan, records):
    if len(records) != plan['evaluation_sources']:
        raise ValueError('heldout report must cover every frozen primary source')
    for task, record in zip(plan['primary_tasks'], records):
        if any(record.get(key) != task[key] for key in ('task_id', 'source_component_id')):
            raise ValueError('heldout record differs from frozen source/task identity')
        tests = record.get('tests')
        if (not isinstance(tests, list) or len(tests) != 10
                or [test.get('test_id') for test in tests] != [str(i) for i in range(10)]):
            raise ValueError('heldout record must contain exactly the original ten distinct test IDs')
        for index, test in enumerate(tests):
            _infrastructure_check(test)
            if type(test.get('hidden')) is not bool or test['hidden'] != (index >= 4):
                raise ValueError('heldout record has incorrect public/private designation')
        passes = [test['outcome'] == 'PASS' for test in tests]
        for key, expected in dict(all_tests_pass=all(passes), visible_all_pass=all(passes[:4]),
                                  hidden_all_pass=all(passes[4:]), test_pass_fraction=sum(passes) / 10).items():
            if record.get(key) != expected or (isinstance(expected, bool) and type(record.get(key)) is not bool):
                raise ValueError('heldout aggregate differs from its raw test outcomes')


def evaluate(case):
    """Score one sealed fresh candidate on four public and six private tests."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    plan, runtime = _phase_plan(case)
    run_receipt, candidates, _ = validate_bank(plan)
    observed = probe_sandbox(plan['pbpf_root'], plan['domain'], plan['timeout_seconds'])
    if observed['execution_lock_sha256'] != runtime['execution_lock_sha256']:
        raise RuntimeError('sandbox environment changed before heldout evaluation')
    private = {row['task_id']: row for row in read_jsonl(Path(plan['evaluator_root']) / 'tasks.jsonl')}
    payloads = [(plan, private[task['task_id']], candidate) for task, candidate in zip(plan['primary_tasks'], candidates)]
    if plan['workers'] == 1:
        records = [_evaluate_candidate(payload) for payload in payloads]
    else:
        # spawn avoids mixing preexec resource limits with threaded fork state.
        with ProcessPoolExecutor(max_workers=plan['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
            records = list(pool.map(_evaluate_candidate, payloads, chunksize=1))
    _records_contract(plan, records)
    mapping = {row['source_component_id']: row for row in read_jsonl(plan['eval_source_id_file_path'])}
    for record in records:
        original = mapping[record['source_component_id']]
        if original.get('problem_id') is not None:
            record['problem_id'] = original['problem_id']
    observed = probe_sandbox(plan['pbpf_root'], plan['domain'], plan['timeout_seconds'])
    if observed['execution_lock_sha256'] != runtime['execution_lock_sha256']:
        raise RuntimeError('sandbox environment changed during heldout evaluation')
    check_frozen(plan, Path(case))
    validate_bank(plan)
    count = len(records)
    all_passes = sum(record['all_tests_pass'] for record in records)
    hidden_passes = sum(record['hidden_all_pass'] for record in records)
    visible_passes = sum(record['visible_all_pass'] for record in records)
    report = dict(schema=REPORT_SCHEMA, **PROTOCOL, historical_replication=False,
                  model=plan['model_id'], revision=plan['revision'], domain=plan['domain'], task_kind=plan['task_kind'],
                  original_paper_cell=plan['original_paper_cell'], adapter_sha256=plan['adapter_tree_sha256'],
                  task_manifest_sha256=plan['evaluator_manifest_sha256'], public_manifest_sha256=plan['public_manifest_sha256'],
                  public_tasks_sha256=plan['public_tasks_sha256'], evaluator_tasks_sha256=plan['evaluator_tasks_sha256'],
                  execution_profile='sandbox', execution_policy=runtime['execution_policy'],
                  execution_lock_sha256=runtime['execution_lock_sha256'], generation_policy=plan['generation_policy'],
                  generator_dependency_versions=run_receipt['generator_dependency_versions'],
                  generation_runtime=run_receipt['generation_runtime'], frozen_plan_sha256=plan['frozen_plan_sha256'],
                  generation_bank_sha256=file_sha(Path(case) / 'evaluation/bank/complete.json'),
                  runtime_sha256=plan['runtime_sha256'], sources=count, all_tests_passes=all_passes,
                  visible_passes=visible_passes, hidden_passes=hidden_passes,
                  fresh_all_tests_pass_at_1=all_passes / count, fresh_visible_pass_at_1=visible_passes / count,
                  fresh_hidden_pass_at_1=hidden_passes / count,
                  mean_test_pass_fraction=sum(record['test_pass_fraction'] for record in records) / count,
                  tests=count * 10, test_passes=sum(test['outcome'] == 'PASS' for record in records for test in record['tests']),
                  records=records, completed_at=now())
    output = Path(case) / 'evaluation/heldout'
    output.mkdir(exist_ok=False)
    with (output / 'records.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
    write_json(output / 'report.json', report)
    return report


def run(case):
    """Advance a prepared immutable case, stopping on any integrity/infra error."""
    import traceback
    case = Path(case).resolve()
    output = case / 'evaluation'
    plan = read_json(output / 'plan.json')
    if plan.get('status') != 'prepared':
        raise ValueError('evaluation run requires a fresh prepared plan; no overwrite or automatic resume')
    check_frozen(plan, case)
    if any((output / name).exists() for name in ('runtime.json', 'bank', 'heldout', 'complete.json')):
        raise FileExistsError('prepared case already contains evaluation artifacts')
    plan.update(status='running', started_at=now())
    write_json(output / 'plan.json', plan, exclusive=False)
    try:
        runtime = probe_sandbox(plan['pbpf_root'], plan['domain'], plan['timeout_seconds'])
        runtime = dict(runtime, schema='eesd-rebuttal-matched-runtime-v1', **PROTOCOL,
                       frozen_plan_sha256=plan['frozen_plan_sha256'], observed_at=now())
        write_json(output / 'runtime.json', runtime)
        plan['runtime_sha256'] = file_sha(output / 'runtime.json')
        write_json(output / 'plan.json', plan, exclusive=False)
        run_stage(plan['generation_command'], output / 'generation.log')
        check_frozen(plan, case)
        validate_bank(plan)
        plan['bank_complete_sha256'] = file_sha(output / 'bank/complete.json')
        write_json(output / 'plan.json', plan, exclusive=False)
        run_stage(plan['heldout_command'], output / 'heldout.log')
        check_frozen(plan, case)
        validate_bank(plan)
        _runtime(plan)
        report = read_json(output / 'heldout/report.json')
        if (report.get('schema') != REPORT_SCHEMA or any(report.get(k) != v for k, v in PROTOCOL.items())
                or report.get('model') != plan['model_id'] or report.get('revision') != plan['revision']
                or report.get('domain') != plan['domain'] or report.get('adapter_sha256') != plan['adapter_tree_sha256']
                or report.get('task_manifest_sha256') != plan['evaluator_manifest_sha256']
                or report.get('generation_policy') != plan['generation_policy']
                or report.get('execution_profile') != 'sandbox'
                or report.get('execution_lock_sha256') != runtime['execution_lock_sha256']
                or report.get('sources') != plan['evaluation_sources']
                or report.get('generation_bank_sha256') != plan['bank_complete_sha256']):
            raise ValueError('heldout report differs from frozen evaluation identity')
        _records_contract(plan, report.get('records', []))
        expected_passes = sum(record['all_tests_pass'] for record in report['records'])
        if (report.get('all_tests_passes') != expected_passes
                or report.get('fresh_all_tests_pass_at_1') != expected_passes / plan['evaluation_sources']):
            raise ValueError('heldout report aggregate differs from raw outcomes')
        plan.update(status='finished_with_real_heldout', finished_at=now(),
                    heldout_report_sha256=file_sha(output / 'heldout/report.json'),
                    heldout_all_tests_pass_at_1=report['fresh_all_tests_pass_at_1'],
                    heldout_sources=report['sources'], heldout_count=report['sources'],
                    scientific_status='one fresh greedy candidate per source evaluated by the prospective isolated protocol')
        write_json(output / 'plan.json', plan, exclusive=False)
        write_json(output / 'complete.json', dict(schema='eesd-rebuttal-matched-complete-v1', **PROTOCOL,
                   status=plan['status'], case_sha256=plan['case_sha256'], plan_sha256=file_sha(output / 'plan.json'),
                   frozen_plan_sha256=plan['frozen_plan_sha256'], heldout_report_sha256=plan['heldout_report_sha256'],
                   generation_bank_sha256=plan['bank_complete_sha256'], runtime_sha256=plan['runtime_sha256']))
        return plan
    except BaseException as error:
        plan.update(status='failed', failed_at=now(), error=f'{type(error).__name__}: {error}',
                    scientific_status='incomplete; no valid scientific result')
        write_json(output / 'plan.json', plan, exclusive=False)
        with (output / 'failure.log').open('a', encoding='utf-8') as stream:
            stream.write(traceback.format_exc())
        raise
