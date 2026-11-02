from unittest.mock import patch
from types import SimpleNamespace
import shutil
import pytest
from eesd.recursive_execution import run_test, feedback_for_tests, Outcome
from eesd.strict_execution import run_tests_batched


def test_soft_sandbox_fallback_without_bwrap():
    # Deviation from the original fail-closed gate: on shared GPU hosts bwrap
    # cannot start, so we fall back to a resource-limited soft sandbox instead
    # of crashing. The fallback must still produce a correct PASS outcome.
    with (
        patch("eesd.recursive_execution.shutil.which", return_value=None),
        patch(
            "eesd.recursive_execution._run_soft",
            return_value=Outcome("PASS", "1", "1", ""),
        ) as soft,
    ):
        result = run_test("print(1)", {"input": "", "expected": "1"})
        assert result.status == "PASS"
        soft.assert_called_once()


def test_valid_stdout_comparison_and_no_network_mount():
    with (
        patch("eesd.recursive_execution.shutil.which", return_value="/usr/bin/bwrap"),
        patch(
            "eesd.recursive_execution.subprocess.run",
            return_value=SimpleNamespace(stdout="3\n", stderr="", returncode=0),
        ) as mocked,
    ):
        result = run_test("print(3)", {"input": "", "expected": "3\n"})
        assert result.status == "PASS"
        cmd = mocked.call_args.args[0]
        assert "--unshare-all" in cmd and "--bind" not in cmd
        assert "python3" in cmd[-5]


def test_bwrap_startup_error_falls_back_to_soft():
    # bwrap exists but cannot mount namespaces on shared GPU hosts. The runner
    # must fall back to the soft sandbox and record the deviation, not crash.
    with (
        patch("eesd.recursive_execution.shutil.which", return_value="/usr/bin/bwrap"),
        patch(
            "eesd.recursive_execution._run_bwrap",
            side_effect=RuntimeError(
                "bubblewrap startup error: bwrap: Cannot mount proc"
            ),
        ),
        patch(
            "eesd.recursive_execution._run_soft",
            return_value=Outcome("PASS", "3", "3", ""),
        ) as soft,
    ):
        result = run_test("print(3)", {"input": "", "expected": "3"})
        assert result.status == "PASS"
        soft.assert_called_once()
        assert (
            __import__(
                "eesd.recursive_execution", fromlist=["SANDBOX_MODE"]
            ).SANDBOX_MODE
            == "soft"
        )


def test_feedback_requires_distinct_four_tests():
    with pytest.raises(ValueError):
        feedback_for_tests("print(0)", [{"id": 0}] * 4)


def test_feedback_requires_distinct_four_tests():
    with pytest.raises(ValueError):
        feedback_for_tests("print(0)", [{"id": 0}] * 4)


def test_batch_harness_spawns_independent_child_processes():
    # The batch harness must run each test in its own child python process,
    # never exec() the candidate in the harness process (which would share
    # builtins/import/environment state across tests).
    from eesd.recursive_execution import _BATCH_HARNESS
    assert "subprocess.Popen" in _BATCH_HARNESS
    assert "exec(" not in _BATCH_HARNESS
    # The child command receives the candidate code and its own input only;
    # the expected output is compared by the harness, never passed to the child.
    assert "[python, \"-I\", \"-S\", \"-B\", \"-c\", code]" in _BATCH_HARNESS
    assert "start_new_session=True" in _BATCH_HARNESS  # killpg on timeout
    assert "os.killpg" in _BATCH_HARNESS


_bwrap_available = shutil.which("bwrap") is not None


@pytest.mark.skipif(not _bwrap_available, reason="bwrap not available")
def test_batch_matches_sequential_execution():
    code = "import sys\nx=int(sys.stdin.read())\nprint(x*x)"
    tests = [{"input": str(n), "expected": str(n * n)} for n in range(8)]
    tests[3]["expected"] = "999"  # one deliberately wrong oracle
    batch = run_tests_batched(code, tests, timeout=2)
    seq = [run_test(code, t, timeout=2) for t in tests]
    assert [r.status for r in batch] == [r.status for r in seq]
    assert [r.actual for r in batch] == [r.actual for r in seq]


@pytest.mark.skipif(not _bwrap_available, reason="bwrap not available")
def test_batch_isolates_builtins_mutation_across_tests():
    # Test 1 poisons builtins.print; test 2 must run in a fresh child and still
    # print normally. Under the old exec() harness test 2 would inherit the
    # poisoned builtins.
    code = (
        "import builtins, sys\n"
        "if sys.stdin.read().strip() == '1':\n"
        "    builtins.print = lambda *a, **k: None\n"
        "print('hello')"
    )
    rs = run_tests_batched(
        code,
        [{"input": "1", "expected": "hello"}, {"input": "2", "expected": "hello"}],
        timeout=2,
    )
    assert rs[0].status == "WRONG_OUTPUT"  # poisoned, prints nothing
    assert rs[1].status == "PASS"  # fresh child, prints hello


@pytest.mark.skipif(not _bwrap_available, reason="bwrap not available")
def test_batch_isolates_environment_mutation_across_tests():
    code = (
        "import os, sys\n"
        "os.environ['CNT'] = str(int(os.environ.get('CNT', '0')) + 1)\n"
        "print(os.environ['CNT'])"
    )
    rs = run_tests_batched(
        code, [{"input": "", "expected": "1"}, {"input": "", "expected": "1"}], timeout=2
    )
    # Each test starts from CNT=0 in its own process => always prints "1".
    assert [r.status for r in rs] == ["PASS", "PASS"]


@pytest.mark.skipif(not _bwrap_available, reason="bwrap not available")
def test_batch_times_out_forked_grandchild():
    # A forked grandchild that inherits the stdout pipe would make a plain
    # subprocess.run(...).communicate() block forever. start_new_session +
    # killpg must reap the whole group and return TIMEOUT promptly.
    code = (
        "import os, sys\n"
        "c = os.fork()\n"
        "if c == 0:\n"
        "    import time; time.sleep(60)\n"
        "    sys.exit(0)\n"
        "print('done')"
    )
    rs = run_tests_batched(code, [{"input": "", "expected": "done"}], timeout=2)
    assert rs[0].status == "TIMEOUT"
