import os
import sys
import time
import signal
import hashlib
import subprocess
import tempfile
import zipfile
import json
import shutil
import base64
from pathlib import Path


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _group_exists(pgid):
    """Check live non-zombie members of the original session/process group."""
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            tail = stat[stat.rfind(")") + 2:].split()
            state, pgrp, session = tail[0], int(tail[2]), int(tail[3])
            if pgrp == pgid and session == pgid and state != "Z":
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


import math

def _terminate_group(pgid, grace=2.0):
    if pgid <= 1 or pgid == os.getpid() or pgid == os.getpgrp():
        return False
    if not _group_exists(pgid):
        return True

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return not _group_exists(pgid)
    except OSError:
        return False

    deadline = time.monotonic() + min(max(grace, 0.0), 2.0)
    while time.monotonic() < deadline:
        if not _group_exists(pgid):
            return True
        time.sleep(0.05)

    if not _group_exists(pgid):
        return True

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        return False

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not _group_exists(pgid):
            return True
        time.sleep(0.05)
    return not _group_exists(pgid)


def run_supervisor(probe_path, output_dir, timeout_seconds=150):
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be finite and positive")

    probe_path = Path(probe_path).resolve()
    output_dir = Path(output_dir).resolve()
    supervisor_path = Path(__file__).resolve()

    receipt_path = output_dir / "receipt.json"
    stdout_path = output_dir / "stdout.txt"
    stderr_path = output_dir / "stderr.txt"
    archive_path = output_dir / "recovery.zip"
    copied_probe = output_dir / probe_path.name
    copied_supervisor = output_dir / supervisor_path.name
    artifacts = output_dir / "probe_artifacts"

    receipt = {
        "status": "failed",
        "returncode": None,
        "timeout": False,
        "leader_unreaped": False,
        "group_cleanup_ok": None,
        "archive": None,
        "archive_sha256": None,
        "archive_bytes": None,
        "archive_base64": None,
        "error": None,
    }
    process = None
    pgid = None

    def recover_archive():
        if archive_path.exists():
            archive_path.unlink()
        with zipfile.ZipFile(archive_path, "w",
                             compression=zipfile.ZIP_DEFLATED) as z:
            for item in output_dir.rglob("*"):
                if not item.is_file() or item == archive_path:
                    continue
                z.write(item, item.relative_to(output_dir))

        data = archive_path.read_bytes()
        receipt["archive"] = archive_path.name
        receipt["archive_bytes"] = len(data)
        receipt["archive_sha256"] = hashlib.sha256(data).hexdigest()
        receipt["archive_base64"] = __import__("base64").b64encode(
            data
        ).decode("ascii")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts.mkdir(parents=True, exist_ok=True)

        if probe_path == supervisor_path:
            raise ValueError("probe cannot be supervisor source")
        if not probe_path.is_file():
            raise FileNotFoundError(probe_path)
        if copied_probe == copied_supervisor:
            raise ValueError("probe/supervisor destination collision")

        shutil.copy2(probe_path, copied_probe)
        shutil.copy2(supervisor_path, copied_supervisor)

        receipt["probe_sha256"] = _sha256(copied_probe)
        receipt["supervisor_sha256"] = _sha256(copied_supervisor)
        if receipt["probe_sha256"] != _sha256(probe_path):
            raise RuntimeError("probe copy hash mismatch")
        if receipt["supervisor_sha256"] != _sha256(supervisor_path):
            raise RuntimeError("supervisor copy hash mismatch")

        env = os.environ.copy()
        env["TMPDIR"] = str(artifacts)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        with open(stdout_path, "wb") as stdout_file, \
             open(stderr_path, "wb") as stderr_file:
            process = subprocess.Popen(
                [sys.executable, str(copied_probe)],
                cwd=str(output_dir),
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
            pgid = process.pid
            deadline = time.monotonic() + timeout_seconds

            while process.poll() is None:
                if time.monotonic() >= deadline:
                    receipt["timeout"] = True
                    break
                time.sleep(0.05)

            if receipt["timeout"]:
                receipt["group_cleanup_ok"] = _terminate_group(
                    pgid, grace=2.0
                )
                try:
                    process.wait(timeout=2.0)
                    receipt["leader_unreaped"] = False
                except subprocess.TimeoutExpired:
                    receipt["leader_unreaped"] = True
            else:
                receipt["returncode"] = process.returncode

                if _group_exists(pgid):
                    receipt["group_cleanup_ok"] = _terminate_group(
                        pgid, grace=2.0
                    )
                else:
                    receipt["group_cleanup_ok"] = True

                try:
                    process.wait(timeout=2.0)
                    receipt["leader_unreaped"] = False
                except subprocess.TimeoutExpired:
                    receipt["leader_unreaped"] = True

        if receipt["leader_unreaped"]:
            raise RuntimeError("leader remained unreaped after bounded wait")
        if receipt["group_cleanup_ok"] is False:
            raise RuntimeError("process group survived bounded SIGKILL cleanup")

        receipt["status"] = (
            "passed" if receipt["returncode"] == 0
            and not receipt["timeout"] else "failed"
        )

    except Exception as exc:
        receipt["status"] = "failed"
        receipt["error"] = repr(exc)

        if process is not None and pgid is not None:
            if process.poll() is None or _group_exists(pgid):
                receipt["group_cleanup_ok"] = _terminate_group(
                    pgid, grace=2.0
                )
            try:
                process.wait(timeout=2.0)
                receipt["leader_unreaped"] = False
            except subprocess.TimeoutExpired:
                receipt["leader_unreaped"] = True

    finally:
        try:
            _save_json_atomic(receipt_path, receipt)
        except Exception as exc:
            receipt["receipt_write_error"] = repr(exc)

        try:
            recover_archive()
        except Exception as exc:
            receipt["archive_error"] = repr(exc)
            try:
                _save_json_atomic(receipt_path, receipt)
                recover_archive()
            except Exception as recovery_exc:
                receipt["archive_recovery_error"] = repr(recovery_exc)

        try:
            _save_json_atomic(receipt_path, receipt)
        except Exception as exc:
            receipt["receipt_final_write_error"] = repr(exc)

    if receipt.get("archive"):
        print("ARCHIVE_BYTES:", receipt["archive_bytes"])
        print("ARCHIVE_SHA256:", receipt["archive_sha256"])
        print("ARCHIVE_BASE64:", receipt["archive_base64"])
    else:
        print("ARCHIVE_UNAVAILABLE")

    return receipt