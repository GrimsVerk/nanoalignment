"""OpenThoughts-Agent-v1-RL tasks as an RL family for GRPO (bash-c0-1).

Source: open-thoughts/OpenThoughts-Agent-v1-RL, HF revision pinned below. No licence is
stated on the card, so (owner ruling 2026-11-20) it is used by Hugging Face id only and
never rehosted: DATA_DIR is gitignored. Tasks were written by GPT-5 Mini and filtered by
GPT-5-Codex (docs/bash-datasets.md).

Each task: seed files, an instruction, a reference command and the expected captured
output. Conversion:
- task text = the task's Goal paragraph plus its output-path line, with /workspace -> /work
  and /output/command_capture.txt -> /work/output/command_capture.txt (our sandbox);
- the reference command is run twice in our sandbox on fresh seeds; a task is kept only if
  both runs agree with each other AND with the shipped expected output (so the check means
  the same thing here as in the original Docker image), the output is non-empty, the seeds
  are small text files, and the task passes the theme screen and hold-out text check;
- split by a hash of the original NL task: 10% validation, 10% test.
The hidden grader compares /work/output/command_capture.txt with the expected output
byte for byte, like the task's own test.sh.

  uv run python -m data.ot_rl [--download]     # writes DATA_DIR/open/ot_rl.json
"""

import argparse
import hashlib
import io
import json
import re
import tarfile
import urllib.request
from collections import Counter

from common import env
from data.bash_open import SCREEN, _norm

PIN = dict(repo="open-thoughts/OpenThoughts-Agent-v1-RL", revision="39ab71434e90d8f87d2cd69c13b6d8a0cb2c238f",
           file="tasks.parquet", licence="none stated")  # fmt: skip
OUT_REL = "output/command_capture.txt"
SPLITS = (("val", 0.10), ("test", 0.10))


def split_of(key: str) -> str:
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 16**8
    acc = 0.0
    for name, frac in SPLITS:
        acc += frac
        if h < acc:
            return name
    return "train"


def path() -> "env.Path":
    return env.path("DATA_DIR") / "open" / "ot_rl.json"


def _task_text(instr: str) -> str | None:
    goal = re.search(r"## Goal\s*\n(.+?)\n\s*\n", instr, re.S)
    req = re.search(r"^- (The combined output must be written to .+)$", instr, re.M)
    if not goal or not req:
        return None
    t = " ".join(goal.group(1).split()) + " " + req.group(1).replace("`", "")
    return t.replace("/output/command_capture.txt", "/work/" + OUT_REL).replace("/workspace", "/work")


def _run_ref(sb, seeds: dict[str, str], cmd: str):
    """Run the reference command on fresh seeds the way solve.sh does; return the captured
    output, or None on a timeout."""
    from harness import sandbox as sbx

    quoted = "'" + cmd.replace("'", "'\"'\"'") + "'"
    d = sbx.materialize(seeds)
    try:
        r = sb.run(f"mkdir -p output\nbash -c {quoted} > {OUT_REL} 2>&1\ncat {OUT_REL}", d)
        return None if r.timed_out else r.out
    finally:
        sbx.cleanup(d)


def build() -> dict:
    import pyarrow.parquet as pq

    from harness import sandbox as sbx
    from posttrain import holdout

    sb = sbx.get("bwrap")
    grams = holdout.eval_ngrams()
    stats: Counter = Counter()
    tasks = []
    rows = pq.read_table(env.path("DATA_DIR") / "open" / "ot_rl" / PIN["file"]).to_pylist()
    for row in rows:
        stats["raw"] += 1
        tf = tarfile.open(fileobj=io.BytesIO(row["task_binary"]))
        files = {m.name: m for m in tf.getmembers() if m.isfile()}
        try:
            instr = tf.extractfile(files["instruction.md"]).read().decode()
            man = json.loads(tf.extractfile(files["task_manifest.json"]).read())
            expected = tf.extractfile(files["tests/expected_output.txt"]).read().decode()
        except (KeyError, UnicodeDecodeError, ValueError):
            stats["unreadable"] += 1
            continue
        seeds, ok = {}, True
        for name, m in files.items():
            if name.startswith("environment/seeds/"):
                rel = name[len("environment/seeds/") :]
                if m.size > 100_000:
                    ok = False
                    break
                try:
                    seeds[rel] = tf.extractfile(m).read().decode()
                except UnicodeDecodeError:
                    ok = False
                    break
        if not ok:
            stats["seeds_binary_or_large"] += 1
            continue
        task = _task_text(instr)
        if task is None:
            stats["no_goal"] += 1
            continue
        cmd = man.get("mutated_bash", "")
        if SCREEN.search(task) or SCREEN.search(cmd) or any(SCREEN.search(k) for k in seeds):
            stats["screened_theme"] += 1
            continue
        if holdout.text_problems(task + " " + cmd, grams):
            stats["holdout"] += 1
            continue
        if not expected.strip():
            stats["empty_expected"] += 1
            continue
        out1, out2 = _run_ref(sb, seeds, cmd), _run_ref(sb, seeds, cmd)
        if out1 is None or out1 != out2:
            stats["nondeterministic"] += 1
            continue
        if out1 != expected:
            stats["differs_from_original_env"] += 1
            continue
        key = "otrl:" + _norm(man.get("original_nl", "") or task)
        tasks.append(
            dict(
                id=row["path"],
                split=split_of(key),
                task=task,
                files=seeds,
                expected=expected,
                key=key,
                ref=cmd,
            )
        )
        stats["kept"] += 1
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(pin=PIN, tasks=tasks)))
    by = Counter(t["split"] for t in tasks)
    return dict(stats=dict(stats), splits=dict(by))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    if a.download:
        d = env.path("DATA_DIR") / "open" / "ot_rl"
        d.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{PIN['repo']}/resolve/{PIN['revision']}/{PIN['file']}"
        (d / PIN["file"]).write_bytes(urllib.request.urlopen(url, timeout=300).read())
    print(json.dumps(build(), indent=1))


if __name__ == "__main__":
    main()
