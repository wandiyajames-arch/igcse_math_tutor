"""Client for the sandbox worker. ``SandboxClient.call`` never raises.

The worker is a fresh interpreter (``python -I -S``) in its own session and
process group, with a minimal environment and a private empty working
directory. A request that exceeds its time budget gets the whole process group
killed (``TIMEOUT``); a garbled or missing reply kills it too
(``WORKER_ERROR``); the next call starts a fresh worker. After a
``RESOURCE_LIMIT`` the worker is recycled so later verdicts never depend on
memory left behind by earlier jobs.
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import selectors
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import weakref
from pathlib import Path

from .verdict import Reason, Verdict, abstain

log = logging.getLogger("src.grading.sandbox")
ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("worker.py")
MAX_REPLY = 64 * 1024


def _site_dirs() -> str:
    dirs = []
    for key in ("purelib", "platlib"):
        p = sysconfig.get_paths().get(key)
        if p and p not in dirs:
            dirs.append(p)
    for p in sys.path:
        if p.endswith(("site-packages", "dist-packages")) and p not in dirs:
            dirs.append(p)
    return os.pathsep.join(dirs)


def _kill_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.wait(timeout=2)
    except Exception:
        pass


class SandboxClient:
    def __init__(self, test_ops: bool = False, startup_timeout: float = 15.0):
        self._lock = threading.Lock()
        self._proc = None
        self._tmp = None
        self._ids = itertools.count(1)
        self._test_ops = test_ops
        self._startup_timeout = startup_timeout
        self.ready_info: dict = {}
        self.restarts = 0
        self._finalizer = weakref.finalize(self, SandboxClient._cleanup, [None, None])

    # -------------------------------------------------------------- lifecycle
    @staticmethod
    def _cleanup(box):
        proc, tmp = box
        if proc is not None:
            _kill_group(proc)
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    def _start(self):
        self._tmp = tempfile.mkdtemp(prefix="igcse-grading-")
        env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"}
        if self._test_ops:
            env["IGCSE_SANDBOX_TEST_OPS"] = "1"
        self._proc = subprocess.Popen(
            [sys.executable, "-I", "-S", str(WORKER), "--site-packages", _site_dirs(), "--root", str(ROOT)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=self._tmp, env=env, close_fds=True, start_new_session=True)
        self._finalizer.detach()
        self._finalizer = weakref.finalize(self, SandboxClient._cleanup, [self._proc, self._tmp])
        threading.Thread(target=self._drain, args=(self._proc,), daemon=True).start()
        line = self._readline(time.monotonic() + self._startup_timeout)
        if line is None:
            raise RuntimeError("worker did not start")
        self.ready_info = json.loads(line)

    def _drain(self, proc):
        for raw in proc.stderr:
            log.warning("worker stderr: %s", raw.decode(errors="replace").rstrip())

    def _stop(self):
        if self._proc is not None:
            _kill_group(self._proc)
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)
        self._proc, self._tmp = None, None
        self._finalizer.detach()

    def close(self):
        with self._lock:
            self._stop()

    def __del__(self):
        try:
            self._stop()
        except Exception:
            pass

    # ----------------------------------------------------------------- calls
    def _readline(self, deadline):
        buf = b""
        fd = self._proc.stdout.fileno()
        sel = selectors.DefaultSelector()
        sel.register(fd, selectors.EVENT_READ)
        try:
            while b"\n" not in buf:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                if not sel.select(remaining):
                    return None
                chunk = os.read(fd, 65536)
                if not chunk:
                    return None
                buf += chunk
                if len(buf) > MAX_REPLY:
                    raise ValueError("reply too large")
            return buf.split(b"\n", 1)[0].decode()
        finally:
            sel.close()

    def call(self, op: str, args: dict, timeout: float = 2.0, lock_timeout: float = 5.0):
        if not self._lock.acquire(timeout=lock_timeout):
            return abstain(Reason.BUSY, "grader busy")
        try:
            if self._proc is None or self._proc.poll() is not None:
                try:
                    if self._proc is not None:
                        self._stop()
                        self.restarts += 1
                    self._start()
                except Exception as e:
                    self._stop()
                    return abstain(Reason.WORKER_ERROR, f"could not start worker: {e}")
            jid = next(self._ids)
            req = json.dumps({"id": jid, "op": op, "args": args}) + "\n"
            if len(req) > 16 * 1024:
                return abstain(Reason.TOO_LONG, "request too large")
            try:
                self._proc.stdin.write(req.encode())
                self._proc.stdin.flush()
                line = self._readline(time.monotonic() + timeout)
            except (BrokenPipeError, ValueError, OSError):
                self._stop()
                return abstain(Reason.WORKER_ERROR, "worker failed")
            if line is None:
                timed_out = self._proc.poll() is None
                self._stop()
                return abstain(Reason.TIMEOUT if timed_out else Reason.WORKER_ERROR,
                               "time limit" if timed_out else "worker exited")
            try:
                reply = json.loads(line)
            except Exception:
                self._stop()
                return abstain(Reason.WORKER_ERROR, "garbled reply")
            if reply.get("id") != jid or "result" not in reply:
                self._stop()
                return abstain(Reason.WORKER_ERROR, reply.get("error", "mismatched reply"))
            if reply.get("resource_limit"):
                self._stop()  # recycle after a memory-limit hit
            return reply["result"]
        finally:
            self._lock.release()
