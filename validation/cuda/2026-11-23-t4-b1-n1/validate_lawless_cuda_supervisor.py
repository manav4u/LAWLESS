import contextlib, hashlib, importlib, json, math, os, subprocess
import sys, tempfile, time, zipfile, base64
from pathlib import Path

S = importlib.import_module("lawless_cuda_supervisor")
B = Path(tempfile.mkdtemp(prefix="lawless_cpu_supervisor_tests_"))
SUMMARY = B / "summary.json"
results = {}

def alive(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()
        return stat[0] != "Z"
    except (OSError, IndexError):
        return False

def fake(name, code):
    p = B / (name + ".py")
    p.write_text("import os,sys,time,signal,subprocess,json\n"+code)
    return p

def test(name, path, timeout=1):
    out = B / ("out_" + name)
    log = B / (name + ".supervisor.log")
    with open(log,"w") as f, contextlib.redirect_stdout(f):
        r = S.run_supervisor(path, out, timeout_seconds=timeout)
    results[name] = r
    SUMMARY.write_text(json.dumps(results, indent=2))
    assert log.exists() and log.stat().st_size > 0
    assert (out / "receipt.json").exists()
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["status"] == r["status"]

    z = out / "recovery.zip"
    assert z.is_file()
    data = z.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    txt = log.read_text()
    def marker(k):
        m = [x.split(":",1)[1].strip() for x in txt.splitlines()
             if x.startswith(k+":")]
        assert len(m)==1, (name,k)
        return m[0]
    assert int(marker("ARCHIVE_BYTES")) == len(data)
    assert marker("ARCHIVE_SHA256") == digest
    assert base64.b64decode(marker("ARCHIVE_BASE64")) == data
    with zipfile.ZipFile(z) as f:
        names=f.namelist()
        assert any(n.endswith("receipt.json") for n in names)
        if name != "missing":
            assert any("lawless_cuda_supervisor" in n for n in names)
        if name in ("success","ignore","leaderexit"):
            assert any("probe_artifacts/" in n for n in names)
    return out,r

# Success, output markers, partial TMPDIR JSON.
p=fake("success","""print('OUT_MARK',flush=True)
print('ERR_MARK',file=sys.stderr,flush=True)
json.dump({'partial':True},open(os.path.join(os.environ['TMPDIR'],
'partial.json'),'w'))
""")
o,r=test("success",p)
assert r["returncode"]==0 and r["status"]=="passed"
assert any(n.endswith("probe_artifacts/partial.json")
           for n in zipfile.ZipFile(o/"recovery.zip").namelist())
assert "OUT_MARK" in (o/"stdout.txt").read_text()
assert "ERR_MARK" in (o/"stderr.txt").read_text()
assert (o/"probe_artifacts"/"partial.json").exists()

# Exit 7.
p=fake("exit7","print('EXIT7',flush=True);sys.exit(7)")
o,r=test("exit7",p)
assert r["returncode"]==7 and r["status"]=="failed"
assert "EXIT7" in (o/"stdout.txt").read_text()

# Hang.
p=fake("hang","time.sleep(30)")
o,r=test("hang",p)
assert r["timeout"] and r["status"]=="failed"
assert r["group_cleanup_ok"] is True and not r["leader_unreaped"]

# Ignore TERM, descendant also ignores TERM.
p=fake("ignore","""c=subprocess.Popen([sys.executable,'-c',
'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)'])
open(os.path.join(os.environ['TMPDIR'],'pid'),'w').write(str(c.pid))
signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)
""")
o,r=test("ignore",p)
assert r["timeout"] and r["group_cleanup_ok"] is True
pid=int((o/"probe_artifacts"/"pid").read_text())
end=time.monotonic()+1
while alive(pid) and time.monotonic()<end: time.sleep(.05)
assert not alive(pid),f"grandchild survived: {pid}"

# Leader exits, descendant persists.
p=fake("leaderexit","""c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
open(os.path.join(os.environ['TMPDIR'],'pid'),'w').write(str(c.pid))
""")
o,r=test("leaderexit",p)
assert r["returncode"]==0 and r["group_cleanup_ok"] is True
pid=int((o/"probe_artifacts"/"pid").read_text())
end=time.monotonic()+1
while alive(pid) and time.monotonic()<end: time.sleep(.05)
assert not alive(pid),f"descendant survived: {pid}"

# Missing probe must yield failed receipt and recovery.
o,r=test("missing",B/"missing_probe.py")
assert r["status"]=="failed" and r["error"]

# Invalid timeout values.
for v in (0,-1,float("nan"),float("inf"),-float("inf"),True):
    try: S.run_supervisor(p,B/"invalid",timeout_seconds=v)
    except ValueError: pass
    else: raise AssertionError(f"invalid timeout accepted: {v!r}")

# Refuse killing the supervisor's own process group.
assert S._terminate_group(os.getpgrp(),grace=0) is False

SUMMARY.write_text(json.dumps(results,indent=2))
print("ALL ASSERTIONS PASSED; SUMMARY:",SUMMARY)