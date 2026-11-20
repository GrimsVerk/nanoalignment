"""Run untrusted (model-written) commands in a sandbox.

One small interface (`Sandbox.run`) so the backend can be swapped. The default backend is
bubblewrap (owner ruling 2026-11-19): no network, read-only /usr, no /home or /etc, one
throwaway work dir, prlimit limits (address space, processes, file size, CPU time) and a
wall-clock timeout. Measured cost: 2 to 8 ms per call (research-2). Model output is never
run on the host directly.

Mount layout seen by the agent:
  /work       the task directory (read-write, except files bound read-only)
  /opt/tools  harness commands: run_checks, submit (read-only)
  /opt/task   the visible check for this task (read-only)
  /ctl        harness control dir: submit writes the report here
"""

import functools
import os
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

TOOLS_DIR = Path(__file__).resolve().parent / "tools"


@dataclass
class Result:
    rc: int
    out: str
    timed_out: bool


@dataclass
class Limits:
    timeout_s: int = 10
    mem_bytes: int = 1 << 30
    nproc: int = 128
    fsize_bytes: int = 20_000_000
    group_mem_bytes: int = 1 << 30  # cap for the whole command (all its processes), via a systemd scope
    max_out_bytes: int = 256_000  # stdout+stderr kept; more than this and the command is killed


@functools.cache
def scope_prefix() -> tuple[str, ...]:
    """`systemd-run --user --scope` prefix if transient user scopes work here, else () (CI)."""
    if not shutil.which("systemd-run"):
        return ()
    base = ("systemd-run", "--user", "--scope", "--quiet", "--collect")
    try:
        ok = subprocess.run([*base, "true"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    if not ok:
        print(
            "[sandbox] warning: no systemd user scopes; commands are capped per process only", file=sys.stderr
        )
    return base if ok else ()


class Sandbox(Protocol):
    def run(
        self,
        cmd: str,
        work: Path,
        *,
        task_dir: Path | None = None,
        ctl_dir: Path | None = None,
        ro_files: list[str] | None = None,
        stdin: str | None = None,
    ) -> Result: ...


class BwrapSandbox:
    def __init__(self, limits: Limits | None = None):
        self.limits = limits or Limits()

    def argv(self, cmd: str, work: Path, task_dir=None, ctl_dir=None, ro_files=None) -> list[str]:
        lim = self.limits
        a = [
            "bwrap",
            "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib", "/lib64",
            "--symlink", "usr/bin", "/sbin",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--ro-bind", str(TOOLS_DIR), "/opt/tools",
            "--bind", str(work), "/work",
        ]  # fmt: skip
        for rel in ro_files or []:  # control twin: protected files cannot be changed or removed
            if (work / rel).exists():
                a += ["--ro-bind", str(work / rel), f"/work/{rel}"]
        if task_dir:
            a += ["--ro-bind", str(task_dir), "/opt/task"]
        if ctl_dir:
            a += ["--bind", str(ctl_dir), "/ctl"]
        a += [
            "--chdir", "/work",
            "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
            "--setenv", "PATH", "/opt/tools:/usr/bin",
            "--setenv", "HOME", "/work",
            "--setenv", "LANG", "C.UTF-8",
            "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            "--hostname", "sandbox",
            "--",
            "prlimit", f"--as={lim.mem_bytes}", f"--nproc={lim.nproc}", f"--fsize={lim.fsize_bytes}",
            f"--cpu={lim.timeout_s + 2}", "--",
            "bash", "-c", cmd,
        ]  # fmt: skip
        pre = scope_prefix()
        if pre:
            pre = (*pre, "-p", f"MemoryMax={lim.group_mem_bytes}", "-p", "MemorySwapMax=0",
                   "-p", f"TasksMax={lim.nproc * 2}", "--")  # fmt: skip
        return [*pre, *a]

    def run(self, cmd, work, *, task_dir=None, ctl_dir=None, ro_files=None, stdin=None) -> Result:
        args = self.argv(cmd, Path(work), task_dir, ctl_dir, ro_files)
        lim = self.limits
        p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if stdin:
            try:
                p.stdin.write(stdin.encode())  # small (submit reports); before reading is fine
            except BrokenPipeError:
                pass
        p.stdin.close()
        bufs = {p.stdout: bytearray(), p.stderr: bytearray()}
        sel = selectors.DefaultSelector()
        for f in bufs:
            sel.register(f, selectors.EVENT_READ)
        deadline = time.monotonic() + lim.timeout_s
        timed_out = flooded = False
        while sel.get_map():
            left = deadline - time.monotonic()
            if left <= 0:
                timed_out = True
                break
            for key, _ in sel.select(timeout=min(left, 1.0)):
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    sel.unregister(key.fileobj)
                    continue
                b = bufs[key.fileobj]
                b += chunk[: max(0, lim.max_out_bytes - len(b))]
            if sum(len(b) for b in bufs.values()) >= lim.max_out_bytes:
                flooded = True
                break
        sel.close()
        if timed_out or flooded:
            p.kill()  # bwrap --die-with-parent takes the whole sandbox down with it
        try:
            rc = p.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:  # output closed but the command still runs
            timed_out = True
            p.kill()
            rc = p.wait()
        for f in bufs:
            f.close()
        out = "".join(bytes(b).decode("utf-8", "replace") for b in bufs.values())
        if flooded:
            return Result(rc, out + "\n[output limit reached, command stopped]", False)
        if timed_out:
            return Result(124, out + "\n[timed out]", True)
        if rc in (-9, -15):  # the scope's group cap (memory or tasks) stopped it
            return Result(rc, out + "\n[killed: memory or process limit]", False)
        return Result(rc, out, False)


def available() -> bool:
    """True if bwrap exists and can create its namespaces here (CI runners may forbid it)."""
    if not shutil.which("bwrap"):
        return False
    d = new_dir()
    try:
        return BwrapSandbox().run("true", d).rc == 0
    finally:
        cleanup(d)


def get(backend: str = "bwrap", limits: Limits | None = None) -> Sandbox:
    if backend == "bwrap":
        return BwrapSandbox(limits)
    raise ValueError(f"unknown sandbox backend {backend!r} (only bwrap is implemented)")


# ---- work-dir helpers (host side: write and read files, never execute them)


def new_dir(prefix: str = "nasb_") -> Path:
    from common.env import ROOT, load

    load()
    base = os.environ.get("SANDBOX_TMP") or None
    if base:  # relative paths are relative to the repo; on disk, not tmpfs, so files are not RAM
        base = ROOT / base
        base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=base))


def materialize(files: dict[str, str], d: Path | None = None) -> Path:
    d = d or new_dir()
    for rel, content in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return d


def snapshot(d: Path, max_bytes: int = 200_000, budget: int = 20_000_000, first=()) -> dict[str, str]:
    """Read back a work dir as {relpath: text}. Files in `first` (the task's own files) are
    read first; after `budget` bytes the rest are listed as "<skipped>", so a model that
    writes thousands of files cannot blow up the host process (bash-c0-1)."""
    paths: list[Path] = []
    for root, _, files in os.walk(d):
        paths += [Path(root) / fn for fn in files]
    pri = set(first)
    paths.sort(key=lambda p: str(p.relative_to(d)) not in pri)
    out: dict[str, str] = {}
    used = 0
    for p in paths:
        rel = str(p.relative_to(d))
        if p.is_symlink():
            out[rel] = "<symlink>"
            continue
        if used >= budget:
            out[rel] = "<skipped>"
            continue
        try:
            with open(p, "rb") as f:
                b = f.read(max_bytes)
            out[rel] = b.decode("utf-8", "replace")
            used += len(b)
        except OSError:
            out[rel] = "<unreadable>"
    return out


def cleanup(d: Path) -> None:
    for root, dirs, files in os.walk(d):  # undo chmod 000 used by some control twins
        for n in dirs + files:
            try:
                os.chmod(Path(root) / n, 0o700)
            except OSError:
                pass
    shutil.rmtree(d, ignore_errors=True)
