"""CPU-only eligibility and immutable scientific training-outcome protocol."""
import hashlib
import json
import math
import re
from pathlib import Path
from eesd.distillation import TRAIN_RULES,validate_scored_record

NON_ESTIMABLE='non_estimable_zero_positive_training_weight'
PROTOCOL_ID='eesd-zero-positive-training-weight-v1'
ADOPTION='docs/EESD_ZERO_WEIGHT_ADOPTION_20260920.md'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def protocol_binding(root):
    return dict(training_outcome_protocol=PROTOCOL_ID,training_outcome_adoption_sha256=sha(Path(root)/ADOPTION),
        training_outcomes_source_sha256=sha(Path(__file__)))


def tree_sha(directory):
    root=Path(directory);files=sorted(p for p in root.rglob('*') if p.is_file())
    if not files:raise ValueError('parent adapter tree must be nonempty')
    digest=hashlib.sha256()
    for p in files:
        name=p.relative_to(root).as_posix().encode();digest.update(len(name).to_bytes(4,'big'));digest.update(name)
        digest.update(bytes.fromhex(sha(p)))
    return digest.hexdigest()


def read_training_rows(path,rule):
    if rule not in TRAIN_RULES or rule=='no_update':raise ValueError('valid update rule required')
    rows=[];ids=set();sources={}
    # JSONL is delimited ONLY by literal LF. Python splitlines() also
    # treats Unicode NEL (U+0085) inside legitimate task text as a separator,
    # silently splitting one JSON record into invalid fragments.
    for line in Path(path).read_text().split("\n"):
        if not line.strip():continue
        row=json.loads(line)
        if not isinstance(row,dict):raise ValueError('scored record must be an object')
        weights=row.get('training_weights')
        if not isinstance(weights,dict) or any(type(w) not in (int,float) for w in weights.values()):
            raise ValueError('training weights must be numeric, not booleans or strings')
        validate_scored_record(row)
        split=row['split']
        if split not in {'train','development'}:raise ValueError('training split must be train/development')
        for key in ('trajectory_id','source_component_id','prompt','correction'):
            if not isinstance(row[key],str) or not row[key].strip():raise ValueError(f'nonempty {key} required')
        if row['trajectory_id'] in ids:raise ValueError('duplicate trajectory identity')
        ids.add(row['trajectory_id'])
        source=row['source_component_id']
        if source in sources and sources[source]!=split:raise ValueError('training/development source overlap')
        sources[source]=split
        before,after,relevance=(row[k] for k in ('before_outcomes','after_outcomes','relevance'))
        if (not all(isinstance(v,list) for v in (before,after,relevance)) or not before
                or len(before)!=len(after) or len(before)!=len(relevance)
                or any(type(x) is not int or not 0<=x<5 for x in before+after)
                or any(type(x) not in (float,int) or not math.isfinite(x) or x<0 for x in relevance)
                or sum(relevance)<=0):raise ValueError('invalid scored trajectory vectors')
        if 'messages' in row:
            messages=row['messages'];metadata=row.get('prompt_metadata',{});token_sha=row.get('prompt_token_ids_sha256')
            if (not isinstance(messages,list) or not messages or any(not isinstance(m,dict)
                    or set(m)!={'role','content'} or m['role'] not in {'system','user'}
                    or not isinstance(m['content'],str) for m in messages)
                    or messages[-1]!={'role':'user','content':row['prompt']}
                    or not isinstance(token_sha,str) or len(token_sha)!=64
                    or any(c not in '0123456789abcdef' for c in token_sha)
                    or not isinstance(metadata,dict) or type(metadata.get('input_tokens')) is not int
                    or metadata['input_tokens']<=0):raise ValueError('invalid saved generation messages/token metadata')
        rows.append({**row,'_weight':float(row['training_weights'][rule])})
    eligibility=classify(rows)
    return rows,eligibility


def classify(rows):
    train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='development']
    if not train or not dev:raise ValueError('nonempty training and development rows required')
    positive=sum(r['_weight']>0 for r in train)
    return dict(status='eligible' if positive else NON_ESTIMABLE,train_rows=len(train),development_rows=len(dev),
        all_rows=len(rows),positive_training_rows=positive,positive_development_rows=sum(r['_weight']>0 for r in dev),
        training_sources=len({r['source_component_id'] for r in train}),
        development_sources=len({r['source_component_id'] for r in dev}))



_GEMMA_TEXT_TARGET = re.compile(
    r"(?:model\.language_model|language_model\.model)\.layers\.[0-9]+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
)


def gemma_text_lora_targets(module_names):
    """Select text decoder projections in Transformers 4.51 and 4.52+ layouts."""
    targets = sorted({name for name in module_names
                      if isinstance(name, str) and _GEMMA_TEXT_TARGET.fullmatch(name)})
    if not targets:
        raise ValueError(
            'Gemma 3 text decoder LoRA targets not found; expected '
            'language_model.model.layers.* (4.51) or model.language_model.layers.* (4.52+)'
        )
    return targets


def adapter_metadata(directory, model_config):
    """Check pinned adapter identity and safetensors structure without ML imports.

    Native tensor decoding still happens in the trainer. This preflight also runs
    on CPU-only hosts preparing cases and rejects missing/truncated checkpoints.
    """
    import yaml
    directory = Path(directory)
    try:
        cfg = json.loads((directory / 'adapter_config.json').read_text())
        model = yaml.safe_load(Path(model_config).read_text())
        if (not isinstance(cfg, dict) or not isinstance(model, dict)
                or cfg.get('peft_type') != 'LORA' or cfg.get('task_type') != 'CAUSAL_LM'
                or cfg.get('base_model_name_or_path') != model.get('model_id')
                or not isinstance(model.get('revision'), str)
                or re.fullmatch(r'[0-9a-f]{40}', model['revision']) is None
                or cfg.get('revision') != model['revision']
                or type(cfg.get('r')) is not int or cfg['r'] < 1):
            raise ValueError('adapter config/model identity mismatch')
        for key, default in [('lora_alpha', None), ('lora_dropout', 0.0)]:
            value = cfg.get(key, default)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'invalid adapter {key}')
        if cfg['lora_alpha'] <= 0 or not 0 <= cfg.get('lora_dropout', 0.0) <= 1:
            raise ValueError('invalid adapter LoRA hyperparameters')
        targets = cfg.get('target_modules')
        if not (isinstance(targets, str) and targets.strip()
                or isinstance(targets, list) and targets
                and all(isinstance(name, str) and name.strip() for name in targets)):
            raise ValueError('adapter target_modules must be nonempty')
        if model['model_id'] == 'google/gemma-3-4b-it':
            if not isinstance(targets, list) or gemma_text_lora_targets(targets) != sorted(set(targets)):
                raise ValueError('Gemma adapter targets must be fully qualified text decoder projections')
        weights = directory / 'adapter_model.safetensors'
        with weights.open('rb') as stream:
            size = weights.stat().st_size
            header_size = int.from_bytes(stream.read(8), 'little')
            if not 2 <= header_size <= min(size - 8, 100_000_000):
                raise ValueError('invalid safetensors header length')
            header = json.loads(stream.read(header_size))
        tensors = {k: v for k, v in header.items() if k != '__metadata__'}
        if not tensors or not any('lora_A' in k for k in tensors) or not any('lora_B' in k for k in tensors):
            raise ValueError('adapter contains no LoRA A/B tensors')
        data_size = size - 8 - header_size
        ranges = []
        widths = {'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2, 'I64': 8,
                  'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}
        for tensor in tensors.values():
            shape, offsets = tensor.get('shape'), tensor.get('data_offsets')
            if (not isinstance(shape, list) or not shape
                    or any(type(n) is not int or n < 1 for n in shape)
                    or not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(n) is not int for n in offsets)
                    or not 0 <= offsets[0] < offsets[1] <= data_size
                    or tensor.get('dtype') not in widths
                    or offsets[1] - offsets[0] != math.prod(shape) * widths[tensor['dtype']]):
                raise ValueError('invalid or truncated adapter tensor')
            ranges.append(offsets)
        cursor = 0
        for start, end in sorted(ranges):
            if start != cursor:
                raise ValueError('overlapping or incomplete adapter tensor storage')
            cursor = end
        if cursor != data_size:
            raise ValueError('unexpected trailing adapter tensor bytes')
        return cfg
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
        raise ValueError(f'invalid parent adapter: {exc}') from exc


def validate_parent_adapter(directory, model_config):
    """Validate a real LoRA checkpoint on CPU before classifying an empty update."""
    import yaml
    from safetensors import safe_open
    directory=Path(directory)
    try:
        cfg=json.loads((directory/'adapter_config.json').read_text())
        model=yaml.safe_load(Path(model_config).read_text())
        if (cfg.get('peft_type')!='LORA' or cfg.get('task_type')!='CAUSAL_LM'
                or cfg.get('base_model_name_or_path')!=model['model_id']
                or cfg.get('revision') not in (None,model['revision'])
                or type(cfg.get('r')) is not int or cfg['r']<1):
            raise ValueError('adapter config/model identity mismatch')
        with safe_open(directory/'adapter_model.safetensors',framework='np') as tensors:
            keys=list(tensors.keys())
            if not keys or not any('lora_' in key for key in keys):
                raise ValueError('adapter contains no LoRA tensors')
            for key in keys:
                if not tensors.get_slice(key).get_shape():raise ValueError('adapter scalar tensor')
    except Exception as exc:
        raise ValueError(f'invalid parent adapter: {exc}') from exc


def zero_report(*,root,input_path,model_config,trainer,rule,seed,budget,previous_adapter,anchor_beta,eligibility):
    if eligibility['status']!=NON_ESTIMABLE:raise ValueError('zero outcome requires verified empty selection')
    if type(budget) is not int or budget<1:raise ValueError('zero outcome requires explicit positive response-token budget')
    previous=str(Path(previous_adapter).resolve()) if previous_adapter else None
    if previous:validate_parent_adapter(previous,model_config)
    return dict(schema='eesd-zero-training-outcome-v1',status=NON_ESTIMABLE,rule=rule,seed=seed,
        input_sha256=sha(input_path),model_config_sha256=sha(model_config),trainer_source_sha256=sha(trainer),
        previous_adapter=previous,previous_adapter_tree_sha256=tree_sha(previous) if previous else None,
        anchor_beta=anchor_beta,response_token_budget=budget,response_tokens=0,optimizer_steps=0,micro_steps=0,
        eligibility=eligibility,adapter=None,main_estimate=None,confidence_interval=None,p_value=None,
        reason='valid training population contains no positive-weight trajectories; equal actual budget is non-estimable',
        descriptive_fallback_adopted=False,**protocol_binding(root))
