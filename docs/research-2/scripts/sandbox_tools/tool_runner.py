"""Structured-tool interface, executed INSIDE the sandbox.

Reads one JSON tool call on stdin, acts on /work, prints the observation.
Tools: list_dir(path), read_file(path), write_file(path, content), run_tests(), submit_report(report)
"""
import json, os, subprocess, sys

raw = sys.stdin.read()
try:
    call = json.loads(raw, strict="--lenient" not in sys.argv)
    if not isinstance(call, dict):
        raise ValueError("not an object")
except Exception as e:  # noqa
    print(f"Error: the action is not valid JSON ({e}). Write one JSON object like "
          '{"tool": "read_file", "path": "notes.txt"}.')
    sys.exit(2)

tool = call.get("tool")
args = call.get("args", call)  # accept both {"tool":..,"path":..} and {"tool":..,"args":{..}}


def safe(p):
    p = os.path.normpath(os.path.join("/work", str(p or ".")))
    if not (p == "/work" or p.startswith("/work/")):
        print("Error: path is outside the task directory.")
        sys.exit(2)
    return p


try:
    if tool == "list_dir":
        p = safe(args.get("path", "."))
        for name in sorted(os.listdir(p)):
            print(name + ("/" if os.path.isdir(os.path.join(p, name)) else ""))
    elif tool == "read_file":
        with open(safe(args.get("path"))) as f:
            sys.stdout.write(f.read())
    elif tool == "write_file":
        p = safe(args.get("path"))
        content = args.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content)
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(p, "w") as f:
            f.write(content)
        print(f"Wrote {len(content)} characters to {os.path.relpath(p, '/work')}.")
    elif tool == "run_tests":
        r = subprocess.run(["/opt/tools/run_tests"], capture_output=True, text=True)
        print((r.stdout + r.stderr).rstrip())
    elif tool == "submit_report":
        with open("/ctl/report", "w") as f:
            f.write(str(args.get("report", "")))
        print("Submitted.")
    else:
        print(f"Error: unknown tool {tool!r}. Tools: list_dir, read_file, write_file, run_tests, submit_report.")
        sys.exit(2)
except FileNotFoundError as e:
    print(f"Error: {e.strerror}: {os.path.relpath(e.filename, '/work') if e.filename else ''}")
    sys.exit(1)
except IsADirectoryError:
    print("Error: that path is a directory.")
    sys.exit(1)
