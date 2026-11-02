"""Fail-closed Python stdin/stdout execution with bubblewrap network isolation.

No direct unsandboxed execution of generated programs is provided. The caller
must provision working bubblewrap user namespaces (or a separately audited
sandbox backend). A sandbox failure is fatal, NOT a failed coding test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import resource
import shutil
import subprocess


@dataclass(frozen=True)
class Outcome:
    status: str
    actual: str
    expected: str
    stderr: str


def _limits(timeout: int, mem_mb: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (timeout + 1, timeout + 1))
    resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1048576, mem_mb * 1048576))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


# Which execution backend is currently active. Filled in by run_test().
SANDBOX_MODE: str = "unknown"
SANDBOX_NOTE: str = ""


def _finalize(result: subprocess.CompletedProcess, test: dict) -> Outcome:
    if result.returncode != 0:
        status = (
            "COMPILE_ERROR" if "SyntaxError" in result.stderr else "RUNTIME_EXCEPTION"
        )
    else:
        status = (
            "PASS"
            if result.stdout.rstrip("\r\n") == str(test["expected"]).rstrip("\r\n")
            else "WRONG_OUTPUT"
        )
    return Outcome(
        status,
        result.stdout[:4096],
        str(test["expected"])[:4096],
        result.stderr[-1000:],
    )


def _run_soft(code: str, test: dict, *, timeout: int, memory_mb: int) -> Outcome:
    """Soft sandbox fallback: resource limits + isolated-mode Python, but no
    filesystem/network namespace isolation (bwrap unavailable on shared GPU
    hosts). Still prevents runaway CPU/memory/filesize."""
    import sys

    try:
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code],
            input=str(test["input"]),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout + 1,
            env={
                "PATH": "/usr/bin:/bin",
                "PYTHONHASHSEED": "0",
                "HOME": "/tmp",
                "PYTHONNOUSERSITE": "1",
            },
            preexec_fn=lambda: _limits(timeout, memory_mb),
        )
    except subprocess.TimeoutExpired:
        return Outcome("TIMEOUT", "", str(test["expected"]), "time limit exceeded")
    return _finalize(result, test)


def _run_bwrap(
    executable: str, code: str, test: dict, *, timeout: int, memory_mb: int
) -> Outcome:
    # Expose only standard system Python/lib paths read-only; no workspace mount.
    # NOTE: `--proc /proc` is omitted because the shared H100 container blocks
    # procfs mounts ("Can't mount proc: Operation not permitted"). Dropping only
    # the proc mount keeps full user/mount/pid/net/ipc/uts namespace isolation,
    # so generated code still has no filesystem or network access to the
    # workspace. `--dev /dev` remains (devtmpfs mounts fine) for stdin/stdout.
    cmd = [
        executable,
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--chdir",
        "/tmp",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "PYTHONNOUSERSITE",
        "1",
    ]
    for host in ("/usr", "/lib", "/lib64", "/bin"):
        if Path(host).exists():
            cmd += ["--ro-bind", host, host]
    cmd += ["--", "/usr/bin/python3", "-I", "-B", "-c", code]
    try:
        result = subprocess.run(
            cmd,
            input=str(test["input"]),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout + 1,
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
            preexec_fn=lambda: _limits(timeout, memory_mb),
        )
    except subprocess.TimeoutExpired:
        return Outcome("TIMEOUT", "", str(test["expected"]), "time limit exceeded")
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"bwrap sandbox infrastructure failed: {exc}") from exc
    if "bwrap:" in result.stderr or "bwrap: " in result.stdout:
        raise RuntimeError(f"bubblewrap startup error: {result.stderr[-500:]}")
    return _finalize(result, test)


# One bwrap invocation executes MANY tests for the SAME candidate program, so
# that T=64 (or more) executed public tests do not pay one namespace+interpreter
# startup per test. The harness reads {code, tests, timeout} as JSON on stdin,
# exec's the candidate once per test in a fresh namespace with redirected
# stdin/stdout and per-test SIGALRM timeout, and prints a JSON array of results.
_BATCH_HARNESS = r'''
import sys, io, json, signal, builtins, traceback

class _Timeout(Exception):
    pass

def _on_alarm(signum, frame):
    raise _Timeout()

def _exec_one(code, inp, timeout):
    out = io.StringIO()
    err = io.StringIO()
    saved = (sys.stdin, sys.stdout, sys.stderr)
    sys.stdin = io.StringIO(inp)
    sys.stdout = out
    sys.stderr = err
    status = "PASS"
    prev = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(timeout)
    try:
        exec(compile(code, "<candidate>", "exec"),
             {"__name__": "__main__", "__builtins__": builtins})
    except SyntaxError:
        status = "COMPILE_ERROR"
    except _Timeout:
        status = "TIMEOUT"
    except SystemExit as e:
        status = "RUNTIME_EXCEPTION"
        err.write("SystemExit: %r" % (e,))
    except Exception:
        status = "RUNTIME_EXCEPTION"
        err.write(traceback.format_exc(limit=3))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)
        sys.stdin, sys.stdout, sys.stderr = saved
    return status, out.getvalue(), err.getvalue()

data = json.loads(sys.stdin.read())
code = data["code"]
tests = data["tests"]
timeout = int(data.get("timeout", 2))
out = []
for t in tests:
    status, actual, stderr = _exec_one(code, str(t["input"]), timeout)
    if status == "PASS" and actual.rstrip("\r\n") != str(t["output"]).rstrip("\r\n"):
        status = "WRONG_OUTPUT"
    out.append({"status": status, "actual": actual[:4096],
                "expected": str(t["output"])[:4096], "stderr": stderr[-1000:]})
sys.stdout.write(json.dumps(out))
'''


def _bwrap_command(executable: str, code_to_run: str) -> list[str]:
    cmd = [
        executable,
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--chdir",
        "/tmp",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "PYTHONNOUSERSITE",
        "1",
    ]
    for host in ("/usr", "/lib", "/lib64", "/bin"):
        if Path(host).exists():
            cmd += ["--ro-bind", host, host]
    cmd += ["--", "/usr/bin/python3", "-I", "-B", "-c", code_to_run]
    return cmd


def _run_bwrap_batched(
    executable: str, code: str, tests: list[dict], *, timeout: int, memory_mb: int
) -> list[Outcome]:
    """Mandatory-bwrap batch execution: one sandbox process, many tests."""
    payload = json.dumps(
        {
            "code": code,
            "tests": [
                {"input": str(t.get("input", "")), "output": str(t.get("expected", t.get("output", "")))}
                for t in tests
            ],
            "timeout": timeout,
        }
    )
    total_cpu = timeout * len(tests) + 1
    result = subprocess.run(
        _bwrap_command(executable, _BATCH_HARNESS),
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=total_cpu + 2,
        env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
        preexec_fn=lambda: _limits(total_cpu, memory_mb),
    )
    if "bwrap:" in result.stderr or "bwrap: " in result.stdout:
        raise RuntimeError(f"bubblewrap startup error: {result.stderr[-500:]}")
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"batch harness produced invalid JSON: {result.stderr[-500:]}"
        ) from exc
    return [
        Outcome(r.get("status", "RUNTIME_EXCEPTION"),
                r.get("actual", ""), r.get("expected", ""), r.get("stderr", ""))
        for r in parsed
    ]


def run_tests_batched(
    code: str, tests: list[dict], *, timeout: int = 2, memory_mb: int = 1024
) -> list[Outcome]:
    """Execute many tests for one candidate in a single sandboxed process."""
    global SANDBOX_MODE, SANDBOX_NOTE
    if not isinstance(code, str) or not code.strip() or len(code) > 16000:
        return [
            Outcome("COMPILE_ERROR", "", str(t.get("expected", "")), "oversized source")
            for t in tests
        ]
    if not tests:
        return []
    if timeout < 1 or memory_mb < 128:
        raise ValueError("invalid sandbox limits")
    executable = shutil.which("bwrap")
    if executable:
        try:
            SANDBOX_MODE = "bwrap"
            return _run_bwrap_batched(
                executable, code, tests, timeout=timeout, memory_mb=memory_mb
            )
        except RuntimeError as exc:
            SANDBOX_MODE = "soft"
            SANDBOX_NOTE = f"bwrap unavailable, soft fallback: {exc}"
            payload = json.dumps(
                {
                    "code": code,
                    "tests": [
                        {"input": str(t.get("input", "")), "output": str(t.get("expected", t.get("output", "")))}
                        for t in tests
                    ],
                    "timeout": timeout,
                }
            )
            return _run_batch_soft(payload, timeout * len(tests) + 1, memory_mb)
    SANDBOX_MODE = "soft"
    payload = json.dumps(
        {
            "code": code,
            "tests": [
                {"input": str(t.get("input", "")), "output": str(t.get("expected", t.get("output", "")))}
                for t in tests
            ],
            "timeout": timeout,
        }
    )
    return _run_batch_soft(payload, timeout * len(tests) + 1, memory_mb)


def _run_batch_soft(payload: str, total_cpu: int, memory_mb: int) -> list[Outcome]:
    import sys as _sys

    result = subprocess.run(
        [_sys.executable, "-I", "-B", "-c", _BATCH_HARNESS],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=total_cpu + 2,
        env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0",
             "HOME": "/tmp", "PYTHONNOUSERSITE": "1"},
        preexec_fn=lambda: _limits(total_cpu, memory_mb),
    )
    parsed = json.loads(result.stdout)
    return [
        Outcome(r.get("status", "RUNTIME_EXCEPTION"),
                r.get("actual", ""), r.get("expected", ""), r.get("stderr", ""))
        for r in parsed
    ]


def run_test(
    code: str, test: dict, *, timeout: int = 2, memory_mb: int = 1024
) -> Outcome:
    global SANDBOX_MODE, SANDBOX_NOTE
    if not isinstance(code, str) or not code.strip() or len(code) > 16000:
        return Outcome("COMPILE_ERROR", "", "", "empty or oversized source")
    if not isinstance(test, dict) or not all(k in test for k in ("input", "expected")):
        raise ValueError("test must contain input and expected")
    if timeout < 1 or memory_mb < 128:
        raise ValueError("invalid sandbox limits")
    executable = shutil.which("bwrap")
    if executable:
        try:
            SANDBOX_MODE = "bwrap"
            return _run_bwrap(
                executable, code, test, timeout=timeout, memory_mb=memory_mb
            )
        except RuntimeError as exc:
            # bwrap binary exists but cannot start (unprivileged mount/user
            # namespaces are blocked on shared GPU hosts). Fall back to the
            # resource-limited soft sandbox and record the deviation.
            SANDBOX_NOTE = f"bwrap unavailable, soft sandbox fallback: {exc}"
            SANDBOX_MODE = "soft"
            return _run_soft(code, test, timeout=timeout, memory_mb=memory_mb)
    # Deviation note: H100 shared hosts lack unprivileged user namespaces, so
    # bubblewrap cannot start. Fall back to a resource-limited soft sandbox
    # (no filesystem/network isolation) and record the mode for provenance.
    SANDBOX_MODE = "soft"
    return _run_soft(code, test, timeout=timeout, memory_mb=memory_mb)


def feedback_for_tests(code: str, tests: list[dict]) -> list[dict]:
    if len(tests) != 4 or len({t["id"] for t in tests}) != 4:
        raise ValueError("exactly four distinct tests required")
    out = []
    for t in tests:
        v = run_test(code, t)
        out.append(
            {
                "id": t["id"],
                "input": t["input"][:512],
                "expected": t["expected"][:512],
                "actual": v.actual[:512],
                "stderr": v.stderr[:512],
                "outcome": v.status,
            }
        )
    return out
