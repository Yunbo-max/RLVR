"""Finite-anchor executable-verifier auditing for dual recursive OPSD.

A generated verifier must implement verify(input_text, candidate_output)->bool.
Ground-truth labels are NEVER passed to generated verifier programs. All
execution is delegated to the existing mandatory bubblewrap boundary.

This module's 'validated' refers ONLY to a disjoint finite-sample audit. It
must not be interpreted as formal correctness or an exact RunBugRun oracle.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import ast
import json
import math
import random
from typing import Callable, Sequence

ALLOWED_IMPORTS = frozenset({"math", "re", "json", "collections", "itertools", "functools", "statistics", "bisect", "heapq", "string", "decimal", "fractions"})
FORBIDDEN_CALLS = frozenset({"eval", "exec", "compile", "open", "input", "__import__", "breakpoint", "globals", "locals", "getattr", "setattr", "delattr", "vars"})
MAX_VERIFIER_CHARS = 10000


@dataclass(frozen=True)
class VerdictCase:
    """One labeled input/output example. `label` is trusted, never a tool input."""
    case_id: str
    input_text: str
    candidate_output: str
    label: bool
    origin: str


@dataclass(frozen=True)
class Audit:
    positives: int
    negatives: int
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    abstain_positive: int
    abstain_negative: int
    balanced_accuracy: float
    false_accept_rate: float
    false_reject_rate: float
    coverage: float

    @property
    def has_both_classes(self) -> bool:
        return self.positives > 0 and self.negatives > 0


def audit_verdicts(cases: Sequence[VerdictCase], verdicts: Sequence[bool | None]) -> Audit:
    if not cases or len(cases) != len(verdicts):
        raise ValueError("nonempty equal-length cases/verdicts required")
    if any(v is not None and type(v) is not bool for v in verdicts):
        raise ValueError("verdicts must be boolean or abstention")
    pos = sum(c.label for c in cases)
    neg = len(cases) - pos
    tp = sum(c.label and v is True for c, v in zip(cases, verdicts))
    fn = sum(c.label and v is False for c, v in zip(cases, verdicts))
    tn = sum(not c.label and v is False for c, v in zip(cases, verdicts))
    fp = sum(not c.label and v is True for c, v in zip(cases, verdicts))
    ap = sum(c.label and v is None for c, v in zip(cases, verdicts))
    an = sum(not c.label and v is None for c, v in zip(cases, verdicts))
    # Abstaining does not count as a correct classification.
    balanced = 0.5 * (tp / pos + tn / neg) if pos and neg else 0.0
    return Audit(pos, neg, tp, tn, fp, fn, ap, an, balanced,
                 fp / neg if neg else 1.0, fn / pos if pos else 1.0,
                 (len(cases)-ap-an)/len(cases))


def admit(audit: Audit, *, max_fpr: float = 0.15, max_fnr: float = 0.25,
          min_per_class: int = 4, min_coverage: float = 1.0) -> bool:
    """Independent gate; do not use reserved audit labels for acceptance."""
    if not 0 <= max_fpr <= 1 or not 0 <= max_fnr <= 1 or not 0 <= min_coverage <= 1:
        raise ValueError("invalid gate thresholds")
    return (audit.positives >= min_per_class and audit.negatives >= min_per_class
            and audit.false_accept_rate <= max_fpr
            and audit.false_reject_rate <= max_fnr
            and audit.coverage >= min_coverage)


def verifier_source_sanity(source: str) -> tuple[bool, str]:
    """Cheap reject-only static checks; NOT a security boundary (bwrap is)."""
    if not isinstance(source, str) or not source.strip() or len(source)>MAX_VERIFIER_CHARS:
        return False, "empty_or_long"
    try:
        tree=ast.parse(source)
    except (SyntaxError, ValueError):
        return False, "syntax"
    defined=[]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if isinstance(node, ast.AsyncFunctionDef):
                return False, "async_not_supported"
            defined.append(node)
        elif isinstance(node, ast.Import):
            if any(n.name.split('.')[0] not in ALLOWED_IMPORTS for n in node.names):
                return False, "unapproved_import"
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or '').split('.')[0] not in ALLOWED_IMPORTS:
                return False, "unapproved_import"
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            if not isinstance(node.value, (ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set)):
                return False, "top_level_dynamic_statement"
        elif isinstance(node, (ast.Expr,)) and isinstance(node.value, ast.Constant):
            pass  # docstring
        else:
            return False, "top_level_executable_statement"
    methods=[node for node in defined if node.name=='verify']
    if len(methods)!=1:
        return False, "requires_one_verify_function"
    args=methods[0].args
    if (len(args.posonlyargs)+len(args.args)!=2 or args.vararg or args.kwarg
        or args.kwonlyargs or args.defaults or args.kw_defaults):
        return False, "verify_requires_two_args"
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            return False, "forbidden_call"
        if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
            return False, "dunder_attribute"
        if isinstance(node, ast.Name) and node.id.startswith('__'):
            return False, "dunder_name"
    return True, "ok"


def _negative_output(expected: str, *, seed: int, case_id: str) -> str:
    """Deterministic perturbation for a finite label audit, not a proof of invalidity.

    Prefer actual failed model outputs; synthetic negatives here are a fallback.
    The audit's labels are relative to the pinned expected-output convention.
    """
    rng=random.Random(int(sha256(f'{seed}:{case_id}:negative'.encode()).hexdigest()[:16],16))
    parts=expected.split()
    candidates=[]
    if len(parts)>1:
        candidates.extend([' '.join(parts[:-1]),' '.join(parts[1:])])
    if parts:
        try:
            n=int(parts[0]);candidates.append(str(n+1)+' '+ ' '.join(parts[1:]))
        except (ValueError,OverflowError):
            pass
        candidates.append(parts[0]+'X '+ ' '.join(parts[1:]))
    candidates.append('WRONG_RESULT_'+str(rng.randrange(10**8)))
    choices=[c for c in candidates if c.strip()!=expected.strip()]
    return rng.choice(choices)


def make_cases(tests: Sequence[dict], *, seed: int, negatives: dict[str, Sequence[str]] | None = None,
               tag: str = '') -> list[VerdictCase]:
    """Balanced oracle-relative positives and independently failing outputs.

    `negatives` may supply real WRONG_OUTPUT from on-policy programs, keyed by
    test ID. Else use a deterministic perturbed output; such synthetic labels
    must be described as canonical-output checks, not full semantic ground truth.
    """
    out=[]
    negatives=negatives or {}
    ids=set()
    for t in tests:
        tid=str(t['id'])
        if tid in ids:
            raise ValueError('duplicate test ID')
        ids.add(tid)
        inp=str(t['input']); expected=str(t['expected'])
        out.append(VerdictCase(f'{tag}{tid}:positive',inp,expected,True,'trusted_expected'))
        opts=[v for v in negatives.get(tid,[]) if isinstance(v,str) and v.strip()!=expected.strip()]
        if opts:
            neg=sorted(set(opts))[0];origin='observed_wrong_output'
        else:
            neg=_negative_output(expected,seed=seed,case_id=tid);origin='synthetic_output_perturbation'
        out.append(VerdictCase(f'{tag}{tid}:negative',inp,neg,False,origin))
    return out


def split_tests(tests: Sequence[dict], *, seed: int, source_id: str,
                train: int, gate: int, solver: int) -> tuple[list[dict],list[dict],list[dict]]:
    """Partition by input identity, before observing any verifier predictions."""
    if min(train,gate,solver)<1 or train+gate+solver>len(tests):
        raise ValueError('insufficient distinct tests for roles')
    ids=[str(t['id']) for t in tests]
    inputs=[str(t['input']) for t in tests]
    if len(set(ids))!=len(ids) or len(set(inputs))!=len(inputs):
        raise ValueError('non-disjoint test identifiers/inputs')
    rng=random.Random(int(sha256(f'{seed}:{source_id}:test-partition'.encode()).hexdigest()[:16],16))
    index=list(range(len(tests)));rng.shuffle(index)
    return ([tests[i] for i in index[:train]],
            [tests[i] for i in index[train:train+gate]],
            [tests[i] for i in index[train+gate:train+gate+solver]])


def _harness(source: str) -> str:
    return source + '''

# Trusted harness appended after the generated verifier; no labels are supplied.
import json as _dv_json
import sys as _dv_sys
_dv_data = _dv_json.loads(_dv_sys.stdin.read())
_dv_pred = []
for _dv_item in _dv_data:
    try:
        _dv_v = verify(_dv_item["input"], _dv_item["candidate_output"])
        _dv_pred.append("1" if _dv_v is True else "0" if _dv_v is False else "?")
    except BaseException:
        _dv_pred.append("?")
_dv_sys.stdout.write(" ".join(_dv_pred))
'''


def execute_verifier(source: str, cases: Sequence[VerdictCase], *,
                     run_test_fn: Callable | None=None, timeout: int=3) -> list[bool | None]:
    """Execute source ONCE for all cases in bwrap. Labels never enter stdin.

    Any runtime failure, invalid source, overlong output or malformed token
    becomes abstention. NEVER execute generated source with local exec/eval.
    """
    ok,_=verifier_source_sanity(source)
    if not cases:
        raise ValueError('empty cases')
    if not ok:
        return [None]*len(cases)
    if run_test_fn is None:
        from eesd.strict_execution import run_test as run_test_fn
    payload=[{'input':x.input_text,'candidate_output':x.candidate_output} for x in cases]
    text=json.dumps(payload,ensure_ascii=False)
    if len(text)>50000:
        raise ValueError('verifier case batch too large; reduce audit batch')
    result=run_test_fn(_harness(source),{'input':text,'expected':'1'},timeout=timeout)
    if result.status not in ('PASS','WRONG_OUTPUT'):
        return [None]*len(cases)
    parts=result.actual.strip().split()
    if len(parts)!=len(cases):
        return [None]*len(cases)
    return [True if x=='1' else False if x=='0' else None for x in parts]


def feedback_mistakes(cases:Sequence[VerdictCase],verdicts:Sequence[bool|None],*,
                      limit:int=8)->list[dict]:
    if len(cases)!=len(verdicts) or limit<1:
        raise ValueError('invalid mismatch specification')
    problems=[]
    for case,v in zip(cases,verdicts):
        if v is not case.label:
            problems.append({'input':case.input_text[:320],
                             'candidate_output':case.candidate_output[:320],
                             'correct_verdict':case.label,
                             'verifier_verdict':'ABSTAIN' if v is None else v,
                             'kind':case.origin})
    return problems[:limit]


def choose_candidate(items:Sequence[tuple[str,Audit]],*,max_fpr:float,max_fnr:float,
                     min_per_class:int)->int|None:
    """Gate using TRAIN-SIDE, held-out-from-repair cases only."""
    valid=[(i,a) for i,(_,a) in enumerate(items) if admit(a,max_fpr=max_fpr,max_fnr=max_fnr,
                                                          min_per_class=min_per_class)]
    if not valid:return None
    return max(valid,key=lambda x:(x[1].balanced_accuracy,x[1].coverage,
                                   -len(items[x[0]][0]),-x[0]))[0]


def safe_feedback_for_solver(source:str,candidate_code:str,tests:Sequence[dict],*,
                             run_program:Callable|None=None,
                             run_verifier:Callable|None=None)->tuple[list[dict],dict]:
    """Generated verifier feedback WITHOUT exposing oracle `expected`.

    Ground truth is used only in a separate diagnostic ledger. It does not
    correct generated-verifier verdicts before teacher conditioning. A false
    verdict can thus harm learning: acceptance gate + fixed-oracle ablation
    and hidden audit are mandatory for scientific interpretation.
    """
    if run_program is None:
        from eesd.strict_execution import run_tests_batched as run_program
    if run_verifier is None:
        run_verifier=execute_verifier
    outcomes=run_program(candidate_code,list(tests))
    if len(outcomes)!=len(tests):
        raise RuntimeError('execution result count mismatch')
    cases=[VerdictCase(str(t['id']),str(t['input']),o.actual,False,'solver') for t,o in zip(tests,outcomes)]
    verdicts=run_verifier(source,cases)
    feedback=[]; mismatches=0
    for t,o,v in zip(tests,outcomes,verdicts):
        gold=o.status=='PASS'
        if v is not None and v!=gold:mismatches+=1
        feedback.append({'id':str(t['id']),'input':str(t['input'])[:256],
                         'actual':o.actual[:256],
                         'program_execution_status':('EXECUTED' if o.status in ('PASS','WRONG_OUTPUT') else o.status),
                         'verifier_verdict':'ABSTAIN' if v is None else ('PASS' if v else 'FAIL')})
    return feedback,{'verifier_disagreements_with_local_oracle':mismatches,
                     'verifier_abstentions':sum(v is None for v in verdicts),
                     'cases':len(tests)}
