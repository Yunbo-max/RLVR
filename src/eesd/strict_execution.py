"""Hard isolation boundary for untrusted generated Python.

Bwrap must start with user/mount/network namespaces. This module deliberately
never invokes the legacy soft-sandbox fallback: resource limits alone provide
no filesystem or network confinement. An execution-infrastructure failure is
fatal, not evidence that the candidate failed a coding test.
"""
from __future__ import annotations
import shutil


def run_test(code: str, test: dict, *, timeout: int = 2, memory_mb: int = 1024):
    if not isinstance(code,str) or not code.strip() or len(code)>16000:
        raise ValueError('invalid or oversized generated source; reject candidate upstream')
    if not isinstance(test,dict) or not all(k in test for k in ('input','expected')):
        raise ValueError('missing trusted test input/expected')
    if timeout<1 or memory_mb<128:
        raise ValueError('invalid resource limits')
    exe=shutil.which('bwrap')
    if exe is None:
        raise RuntimeError('no working bubblewrap: refuse untrusted code execution')
    from eesd.recursive_execution import _run_bwrap
    # _run_bwrap must not fall back to legacy _run_soft.
    return _run_bwrap(exe,code,test,timeout=timeout,memory_mb=memory_mb)


def run_tests_batched(code: str, tests: list[dict], *, timeout: int = 2,
                      memory_mb: int = 1024) -> list:
    """Execute many tests for one candidate in a single mandatory-bwrap sandbox.

    Fatal if bubblewrap is unavailable. No soft-sandbox fallback.
    """
    if not isinstance(code,str) or not code.strip() or len(code)>16000:
        from eesd.recursive_execution import Outcome
        return [Outcome('COMPILE_ERROR','',str(t.get('expected','')),'oversized source')
                for t in tests]
    if not tests:
        return []
    if timeout<1 or memory_mb<128:
        raise ValueError('invalid sandbox limits')
    exe=shutil.which('bwrap')
    if exe is None:
        raise RuntimeError('no working bubblewrap: refuse untrusted code execution')
    from eesd.recursive_execution import _run_bwrap_batched
    try:
        return _run_bwrap_batched(exe,code,tests,timeout=timeout,memory_mb=memory_mb)
    except RuntimeError:
        # A candidate called os._exit()/os.abort(), which kills the whole batch
        # harness. Fall back to per-test bwrap (still isolated, only slower).
        return [run_test(code,t,timeout=timeout,memory_mb=memory_mb) for t in tests]


def feedback_for_tests(code: str, tests: list[dict]) -> list[dict]:
    if len(tests)!=4 or len({t['id'] for t in tests})!=4:
        raise ValueError('four distinct public tests required')
    out=[]
    for t in tests:
        result=run_test(code,t)
        out.append({'id':t['id'],
                    'input':str(t['input'])[:512],
                    'expected':str(t['expected'])[:512],
                    'actual':result.actual[:512],
                    'stderr':result.stderr[:512],
                    'outcome':result.status})
    return out


def audit_vector(code: str, tests: list[dict]) -> list[int]:
    if not tests or len({t['id'] for t in tests})!=len(tests):
        raise ValueError('independent audit probes missing/duplicated')
    if not isinstance(code, str) or not code.strip() or len(code)>16000:
        return [0] * len(tests)  # invalid candidate, not a sandbox failure
    return [int(v.status=='PASS') for v in run_tests_batched(code,tests)]
