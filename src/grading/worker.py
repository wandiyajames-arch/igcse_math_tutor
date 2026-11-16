"""Sandbox worker: ``python -I -S worker.py --site-packages <dirs> --root <repo>``.

Module level imports only the standard library. ``main()`` applies the
resource limits (256 MiB address space, no core files, no file writes) before
any third-party module is imported, then serves JSON-lines requests on stdin.
"""
import json
import os
import resource
import sys

MEMORY_LIMIT = 256 * 1024 * 1024
MAX_REQUEST = 16 * 1024


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))


def _reply(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main(argv):
    _limits()
    site_dirs, root = [], None
    it = iter(argv)
    for a in it:
        if a == "--site-packages":
            site_dirs = [d for d in next(it).split(os.pathsep) if d]
        elif a == "--root":
            root = next(it)
    if root:
        sys.path.insert(0, root)
    sys.path.extend(site_dirs)
    third_party_before = sorted(m for m in sys.modules if m.split(".")[0] in ("sympy", "pydantic"))
    from src.grading.grader import grade  # noqa: E402  (after limits)
    from src.grading.verdict import Reason, abstain
    grade({"type": "expression", "value": "x+1", "variables": ["x"]}, "1+x")  # warm-up
    _reply({"ready": True, "rlimit_as": resource.getrlimit(resource.RLIMIT_AS)[0],
            "third_party_at_limit": third_party_before, "pid": os.getpid()})
    for line in sys.stdin:
        if len(line) > MAX_REQUEST:
            _reply({"id": None, "error": "request too large"})
            continue
        try:
            req = json.loads(line)
            jid, op, args = req["id"], req["op"], req.get("args", {})
        except Exception:
            _reply({"id": None, "error": "bad request"})
            continue
        try:
            if op == "ping":
                result = "pong"
            elif op == "grade":
                result = grade(args["spec"], args["answer"]).to_dict()
            elif op == "_spin" and os.environ.get("IGCSE_SANDBOX_TEST_OPS") == "1":
                while True:
                    pass
            elif op == "_allocate" and os.environ.get("IGCSE_SANDBOX_TEST_OPS") == "1":
                _ = bytearray(1024 * 1024 * 1024)
                result = "allocated"
            elif op == "_write_probe" and os.environ.get("IGCSE_SANDBOX_TEST_OPS") == "1":
                try:
                    with open("probe.txt", "w") as f:
                        f.write("x")
                        f.flush()
                    result = "wrote"
                except OSError as e:
                    result = f"blocked:{e.errno}"
            else:
                _reply({"id": jid, "error": f"unknown op {op!r}"})
                continue
            _reply({"id": jid, "result": result})
        except MemoryError:
            _reply({"id": jid, "result": abstain(Reason.RESOURCE_LIMIT, "memory limit").to_dict(),
                    "resource_limit": True})
        except RecursionError:
            _reply({"id": jid, "result": abstain(Reason.TOO_COMPLEX, "too deep").to_dict()})
        except Exception as e:
            sys.stderr.write(f"worker error: {type(e).__name__}: {e}\n")
            _reply({"id": jid, "result": abstain(Reason.WORKER_ERROR, "worker error").to_dict()})


if __name__ == "__main__":
    main(sys.argv[1:])
