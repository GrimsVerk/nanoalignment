"""Run untrusted (model-written) code in a bubblewrap sandbox.

No network, no host files except a read-only /usr, one throwaway work dir,
CPU time, address-space, file-size and process limits, wall-clock timeout.
Model output is NEVER run on the host directly.
"""
import os, shutil, subprocess, tempfile

TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_tools")


def run(cmd_argv, work_dir, timeout=10, mem_bytes=1 << 30, ctl_dir=None, stdin=None):
    """Run argv inside the sandbox with work_dir mounted read-write at /work.

    Returns (exit_code, combined_output, timed_out).
    """
    args = [
        "bwrap",
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib", "/lib64",
        "--symlink", "usr/bin", "/sbin",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--ro-bind", TOOLS_DIR, "/opt/tools",
        "--bind", work_dir, "/work",
    ]
    if ctl_dir:
        args += ["--bind", ctl_dir, "/ctl"]
    args += [
        "--chdir", "/work",
        "--unshare-all", "--die-with-parent", "--new-session",
        "--clearenv",
        "--setenv", "PATH", "/opt/tools:/usr/bin",
        "--setenv", "HOME", "/work",
        "--setenv", "LANG", "C.UTF-8",
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "--setenv", "PYTHONPATH", "/opt/tools",
        "--hostname", "sandbox",
        "--",
        "prlimit", f"--as={mem_bytes}", "--nproc=128", "--fsize=20000000",
        f"--cpu={timeout + 2}", "--",
    ] + list(cmd_argv)
    try:
        p = subprocess.run(args, capture_output=True, timeout=timeout,
                           input=stdin, text=True, errors="replace")
        out = (p.stdout or "") + (p.stderr or "")
        return p.returncode, out, False
    except subprocess.TimeoutExpired as e:
        out = ""
        for s in (e.stdout, e.stderr):
            if s:
                out += s if isinstance(s, str) else s.decode("utf-8", "replace")
        return 124, out + "\n[timed out]", True


def new_dir():
    base = os.environ.get("SANDBOX_TMP") or None
    return tempfile.mkdtemp(prefix="nasb_", dir=base)


def materialize(files, d=None):
    """Write a {relpath: content} dict into a new (or given) temp dir."""
    d = d or new_dir()
    for rel, content in files.items():
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(content)
    return d


def snapshot(d, max_bytes=200_000):
    """Read back a work dir as {relpath: text}. Host only reads, never runs."""
    out = {}
    for root, _, files in os.walk(d):
        for fn in files:
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, d)
            if os.path.islink(p):
                out[rel] = "<symlink>"
                continue
            try:
                with open(p, "rb") as f:
                    out[rel] = f.read(max_bytes).decode("utf-8", "replace")
            except OSError:
                out[rel] = "<unreadable>"
    return out


def cleanup(d):
    shutil.rmtree(d, ignore_errors=True)
