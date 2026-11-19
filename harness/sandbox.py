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

import os
import shutil
import subprocess
import tempfile
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
        return a

    def run(self, cmd, work, *, task_dir=None, ctl_dir=None, ro_files=None, stdin=None) -> Result:
        args = self.argv(cmd, Path(work), task_dir, ctl_dir, ro_files)
        try:
            p = subprocess.run(
                args,
                capture_output=True,
                timeout=self.limits.timeout_s,
                input=stdin,
                text=True,
                errors="replace",
            )
            return Result(p.returncode, (p.stdout or "") + (p.stderr or ""), False)
        except subprocess.TimeoutExpired as e:
            out = ""
            for s in (e.stdout, e.stderr):
                if s:
                    out += s if isinstance(s, str) else s.decode("utf-8", "replace")
            return Result(124, out + "\n[timed out]", True)


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
    from common.env import load

    load()
    base = os.environ.get("SANDBOX_TMP") or None
    return Path(tempfile.mkdtemp(prefix=prefix, dir=base))


def materialize(files: dict[str, str], d: Path | None = None) -> Path:
    d = d or new_dir()
    for rel, content in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return d


def snapshot(d: Path, max_bytes: int = 200_000) -> dict[str, str]:
    """Read back a work dir as {relpath: text}."""
    out: dict[str, str] = {}
    for root, _, files in os.walk(d):
        for fn in files:
            p = Path(root) / fn
            rel = str(p.relative_to(d))
            if p.is_symlink():
                out[rel] = "<symlink>"
                continue
            try:
                out[rel] = p.read_bytes()[:max_bytes].decode("utf-8", "replace")
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
