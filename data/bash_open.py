"""Open bash datasets, converted to the harness's action format for SFT (bash-c0-1).

Owner ruling 2026-11-20: the licence decides use; provenance is recorded as a possible
confound (docs/DECISIONS.md) and every item keeps a `source` tag. Data is downloaded at a
pinned revision into DATA_DIR/open/ and never rehosted.

Sources so far (single command per task, no execution):
  nl2sh   westenfelder/NL2SH-ALFA (MIT): natural-language task -> one bash command.
  tldr    tldr-pages/tldr (CC BY 4.0, attribution in docs/c0-training.md): each example of
          pages/common and pages/linux -> task = its description, action = its command with
          the {{placeholder}} braces removed.
Each item becomes a one-step episode: the harness prompt with the task, then the command
as the action (loss on the action only). Generation in the harness stops at the closing
fence, so the model is never taught to stop after one command.

Screening: items that touch the eval scenarios' themes (tests, logs, audits, lint, CI,
answers, graders, solutions) are dropped, and the rest go through the hold-out text check.
Splits: by a hash of the normalised command (nl2sh) or page (tldr), so the same command or
page never lands in two splits; validation and test items whose task text also occurs in
train are dropped.

  uv run python -m data.bash_open --download     # fetch pinned files into DATA_DIR/open/
  uv run python -m data.bash_open                # write DATA_DIR/sft/open_{train,val,test}.jsonl
"""

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
import urllib.request
from collections import Counter

from common import env
from harness import episode

PINS = {
    "nl2sh": dict(
        repo="westenfelder/NL2SH-ALFA", revision="a99cb5784cf5c2a42b1cc26c1903d9c3b35206ba",
        files=["train.csv", "test.csv"], licence="MIT",
    ),
    "tldr": dict(
        repo="tldr-pages/tldr", revision="cd6becfd6bbeb8f455db0ff16d46042aee8c2f4c",
        files=["pages/common", "pages/linux"], licence="CC BY 4.0 (pages)",
    ),
}  # fmt: skip
SCREEN = re.compile(
    r"\b(tests?|testing|unittest|pytest|logs?|logging|logfiles?|syslog|journalctl|logrotate|audit\w*|lint\w*|"
    r"ci|answers?|grad(e|er|ers|ing)|solutions?|cheat\w*)\b|\.log\b",
    re.I,
)
SPLITS = (("val", 0.03), ("test", 0.03))  # the rest is train


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def split_of(key: str) -> str:
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 16**8
    acc = 0.0
    for name, frac in SPLITS:
        acc += frac
        if h < acc:
            return name
    return "train"


def download() -> None:
    d = env.path("DATA_DIR") / "open"
    p = PINS["nl2sh"]
    (d / "nl2sh").mkdir(parents=True, exist_ok=True)
    for f in p["files"]:
        url = f"https://huggingface.co/datasets/{p['repo']}/resolve/{p['revision']}/{f}"
        (d / "nl2sh" / f).write_bytes(urllib.request.urlopen(url, timeout=60).read())
    p = PINS["tldr"]
    (d / "tldr").mkdir(parents=True, exist_ok=True)
    url = f"https://codeload.github.com/{p['repo']}/tar.gz/{p['revision']}"
    (d / "tldr" / "tldr.tar.gz").write_bytes(urllib.request.urlopen(url, timeout=120).read())


def nl2sh_items():
    d = env.path("DATA_DIR") / "open" / "nl2sh"
    for f in PINS["nl2sh"]["files"]:
        for row in csv.DictReader(open(d / f, newline="")):
            nl, cmd = row["nl"].strip(), row["bash"].strip()
            if nl and cmd:
                yield dict(
                    source="nl2sh", task=nl[0].upper() + nl[1:], cmd=cmd, split_key="cmd:" + _norm(cmd)
                )


def _tldr_cmd(s: str) -> str:
    s = re.sub(r"\{\{\[([^|\]]+)\|[^\]]*\]\}\}", r"\1", s)  # {{[-v|--verbose]}} -> -v
    return re.sub(r"\{\{(.*?)\}\}", r"\1", s).strip()


def tldr_items():
    tar = tarfile.open(env.path("DATA_DIR") / "open" / "tldr" / "tldr.tar.gz")
    for m in tar.getmembers():
        parts = m.name.split("/")
        if not (m.isfile() and len(parts) == 4 and parts[1] == "pages" and parts[2] in ("common", "linux")):
            continue
        if not parts[3].endswith(".md"):
            continue
        text = tar.extractfile(m).read().decode("utf-8", "replace")
        page = parts[3][:-3]
        desc = None
        for line in io.StringIO(text):
            line = line.rstrip("\n")
            if line.startswith("- "):
                desc = re.sub(r"\[(\w+)\]", r"\1", line[2:]).rstrip(":").strip()
            elif line.startswith("`") and line.endswith("`") and desc:
                cmd = _tldr_cmd(line[1:-1])
                if cmd:
                    yield dict(source="tldr", task=desc, cmd=cmd, split_key=f"page:{parts[2]}/{page}")
                desc = None


def to_record(it: dict) -> dict:
    prompt = episode.HEADER + f"## Your task\nTask: {it['task']}\n\n{episode.OPEN}"
    return dict(source=it["source"], family=f"open/{it['source']}", keys=[it["split_key"]],
                segs=[prompt, it["cmd"] + "\n```"])  # fmt: skip


def build(caps: dict[str, int]) -> dict:
    from posttrain import holdout

    stats: Counter = Counter()
    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    eval_grams = holdout.eval_ngrams()
    for src, items in (("nl2sh", nl2sh_items()), ("tldr", tldr_items())):
        seen: set[tuple[str, str]] = set()
        for it in items:
            stats[(src, "raw")] += 1
            k = (_norm(it["task"]), _norm(it["cmd"]))
            if k in seen:
                stats[(src, "duplicate")] += 1
                continue
            seen.add(k)
            if SCREEN.search(it["task"]) or SCREEN.search(it["cmd"]):
                stats[(src, "screened_theme")] += 1
                continue
            if len(it["cmd"]) > 300 or "\n" in it["cmd"]:
                stats[(src, "too_long")] += 1
                continue
            if holdout.text_problems(it["task"] + " " + it["cmd"], eval_grams):
                stats[(src, "holdout")] += 1
                continue
            by_split[split_of(it["split_key"])].append(it)
    train_tasks = {_norm(i["task"]) for i in by_split["train"]}
    for sp in ("val", "test"):
        keep = [i for i in by_split[sp] if _norm(i["task"]) not in train_tasks]
        stats[("all", f"{sp}_dropped_task_in_train")] += len(by_split[sp]) - len(keep)
        by_split[sp] = keep
    out = env.path("DATA_DIR") / "sft"
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for sp, items in by_split.items():
        rows = []
        for src in ("nl2sh", "tldr"):
            mine = sorted((i for i in items if i["source"] == src), key=lambda i: hashlib.sha1(
                (i["task"] + i["cmd"]).encode()).hexdigest())  # fmt: skip
            cap = caps.get(src) if sp == "train" else None
            rows += mine[:cap] if cap else mine
        with open(out / f"open_{sp}.jsonl", "w") as f:
            for i in rows:
                f.write(json.dumps(to_record(i)) + "\n")
        counts[sp] = dict(Counter(i["source"] for i in rows))
    manifest = dict(
        pins=PINS, counts=counts, stats={f"{a}:{b}": n for (a, b), n in sorted(stats.items())}, caps=caps
    )
    (out / "open_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> None:
    from posttrain import shared_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--tier", default="tier1")
    a = ap.parse_args()
    if a.download:
        download()
    caps = shared_config.resolve(a.tier)["sft"]["data"].get("open", {})
    m = build(caps)
    print(json.dumps({"counts": m["counts"], "stats": m["stats"]}, indent=1))


if __name__ == "__main__":
    main()
