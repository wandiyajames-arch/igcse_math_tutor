"""Sandbox worker tests: limits before import, timeout, recycling, isolation."""
import os
import threading
import time

import pytest

from src.grading.grader import grade_safe
from src.grading.sandbox import SandboxClient
from src.grading.verdict import Reason, Status, Verdict


@pytest.fixture
def client():
    c = SandboxClient(test_ops=True)
    yield c
    pid = c._proc.pid if c._proc else None
    c.close()
    if pid:
        with pytest.raises(ProcessLookupError):
            os.killpg(pid, 0)


def test_limit_applied_before_third_party_import(client):
    assert client.call("ping", {}) == "pong"
    assert client.ready_info["rlimit_as"] == 256 * 1024 * 1024
    assert client.ready_info["third_party_at_limit"] == []


def test_grade_through_the_sandbox(client):
    v = grade_safe({"type": "expression", "value": "7a-6b", "variables": ["a", "b"]}, "2a+3b+5a-9b", client=client)
    assert v.reason is Reason.NOT_SIMPLIFIED


def test_timeout_kills_and_next_call_succeeds(client):
    client.call("ping", {})
    old = client._proc.pid
    t = time.monotonic()
    v = client.call("_spin", {}, timeout=2.0)
    assert isinstance(v, Verdict) and v.reason is Reason.TIMEOUT
    assert 1.9 <= time.monotonic() - t < 3.0
    with pytest.raises(ProcessLookupError):
        os.killpg(old, 0)
    assert client.call("ping", {}) == "pong" and client._proc.pid != old


def test_memory_limit_is_handled_and_worker_recycled(client):
    client.call("ping", {})
    old = client._proc
    r = client.call("_allocate", {})
    assert r["reason"] == "resource_limit"
    assert client._proc is None or client._proc is not old
    assert client.call("ping", {}) == "pong"


def test_file_writes_are_blocked(client):
    assert client.call("_write_probe", {}) == "blocked:27"


def test_production_worker_rejects_test_ops():
    c = SandboxClient()
    try:
        v = c.call("_spin", {}, timeout=1.5)
        assert isinstance(v, Verdict) and v.reason is Reason.WORKER_ERROR
    finally:
        c.close()


def test_concurrent_calls_do_not_cross():
    c = SandboxClient()
    results = []

    def work(k):
        for i in range(20):
            v = grade_safe({"type": "numeric", "value": str(k * 100 + i)}, str(k * 100 + i), client=c)
            results.append(v.status is Status.CORRECT)
    threads = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    c.close()
    assert len(results) == 80 and all(results)


def test_latency_measurements(client):
    t = time.perf_counter()
    c2 = SandboxClient()
    c2.call("ping", {})
    cold = (time.perf_counter() - t) * 1000
    c2.close()
    warm = []
    for _ in range(100):
        t = time.perf_counter()
        grade_safe({"type": "numeric", "value": "8/3"}, "2.67", client=client)
        warm.append((time.perf_counter() - t) * 1000)
    warm.sort()
    print(f"\ncold start {cold:.0f} ms; warm grade median {warm[50]:.2f} ms, p95 {warm[95]:.2f} ms")
    assert cold < 5000 and warm[50] < 100
