"""Write bank.json: many sentence variants per slot, written by a local open model.

Provenance (CoS 2026-11-20, pending the owner's ruling): training-target sentences must
not be written by Claude. Claude wrote only the instructions below and in slots.yaml.
The local model (Gemma 4 12B instruct, Apache 2.0, pinned in configs/pins.yaml) writes
the sentences; code filters them (placeholders, length, characters, report class,
distance from eval task texts) and the same model checks each one keeps the meaning.
Nothing in bank.json is edited by hand.

  uv run python -m harness.templates.paraphrase [--slots a,b] [--per-call 20 --calls 4 --keep 40]
"""

import argparse
import json
import random
import re
import time
from difflib import SequenceMatcher

from common import config, env
from harness import reports
from harness.templates import DIR, PLACEHOLDER, slots

STYLE = {
    "task": "It is an instruction given to someone who works in a Linux terminal.",
    "report_ok": (
        "It is the short final report that the person who did the work writes when they finish. "
        "Write it as that person, in plain words."
    ),
    "report_fail": (
        "It is the short final report that the person who tried the task writes when they finish. "
        "Write it as that person, in plain words, and be clear that it did not work out."
    ),
    "note": "It is a terse note that someone appends to a text file. Keep it to a few words.",
}
PROMPT = (
    "Write {n} different versions of one sentence for a dataset of command-line tasks.\n\n"
    "{style}\n\nEach version must say: {meaning}\n\n"
    "Keep these placeholders exactly as written, with their curly braces, each one at least once: {phs}. "
    "They stand for values, for example {examples}.\n"
    "Vary the wording, the sentence structure and the length. Use plain English. "
    "Do not use double quotes, dollar signs, backslashes or backticks.\n"
    "Output one version per line, numbered 1 to {n}, and nothing else."
)
CHECK = (
    "Here is a sentence:\n\n{sent}\n\nDoes this sentence say all of the following, possibly in other words, "
    "and nothing that contradicts it?\n\n{meaning}\n\nAnswer with one word: yes or no."
)
MAX_WORDS = {"task": 60, "report_ok": 45, "report_fail": 55, "note": 16}
BAD_CHARS = {"task": '`\\"', "report_ok": '"$`\\', "report_fail": '"$`\\', "note": "\"'$`\\"}


def example(slot: dict, text: str) -> str:
    return text.format(**{k: str(v) for k, v in slot["vars"].items()})


def eval_skeletons():
    from harness import scenarios
    from posttrain import holdout

    sks = {
        holdout.skeleton(scenarios.instance(s, v, i).task)
        for s in scenarios.SHORTCUT
        for v in ("open", "control")
        for i in range(5)
    }
    grams = set().union(*(holdout.ngrams(t) for t in sks))
    return sks, grams


def code_filter(name: str, slot: dict, sent: str, ev) -> str | None:
    """None if the sentence passes, else the reason it was dropped."""
    from posttrain import holdout

    kind = slot["kind"]
    if "\n" in sent or not sent:
        return "empty"
    if set(PLACEHOLDER.findall(sent)) != set(slot["vars"]):
        return "placeholders"
    rest = re.sub(r"\{\w+\}", "", sent)
    if "{" in rest or "}" in rest:
        return "braces"
    if any(c in sent for c in BAD_CHARS[kind]):
        return "chars"
    if not 3 <= len(sent.split()) <= MAX_WORDS[kind]:
        return "length"
    ex = example(slot, sent)
    if kind == "report_ok" and not reports.claims_success(ex):
        return "not_success_claim"
    if kind == "report_fail" and (reports.claims_success(ex) or not reports.admits_failure(ex)):
        return "not_failure_report"
    if kind.startswith("report") and reports.is_copied(ex):
        return "generic"
    if kind == "task":
        sks, grams = ev
        sk = holdout.skeleton(ex)
        if max(SequenceMatcher(None, sk, t).ratio() for t in sks) > holdout.MAX_SIMILARITY - 0.05:
            return "near_eval_task"
        if holdout.ngrams(ex) & grams:
            return "eval_ngram"
    return None


def parse(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        m = re.match(r"^\s*\d+[.)]\s*(.+?)\s*$", line)
        if m:
            s = m.group(1).strip().strip("*").strip()
            if len(s) > 1 and s[0] == s[-1] and s[0] in "'‘’“”":
                s = s[1:-1].strip()
            out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", help="comma list (default: all)")
    ap.add_argument("--per-call", type=int, default=20)
    ap.add_argument("--calls", type=int, default=4)
    ap.add_argument("--keep", type=int, default=40)
    ap.add_argument("--gpu-util", type=float, default=0.86)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams

    S = slots()
    names = a.slots.split(",") if a.slots else list(S)
    pin = config.pins()["models"]["paraphraser"]
    t0 = time.time()
    llm = LLM(model=pin["repo"], revision=pin["revision"], max_model_len=4096, max_num_seqs=64,
              gpu_memory_utilization=a.gpu_util, language_model_only=True, compilation_config={"mode": 0},
              trust_remote_code=False, seed=0)  # fmt: skip
    convs, owners = [], []
    for name in names:
        s = S[name]
        phs = ", ".join("{" + k + "}" for k in s["vars"])
        exs = ", ".join(f"{{{k}}} = {v}" for k, v in s["vars"].items())
        msg = PROMPT.format(n=a.per_call, style=STYLE[s["kind"]], meaning=s["meaning"], phs=phs, examples=exs)
        for _ in range(a.calls):
            convs.append([{"role": "user", "content": msg}])
            owners.append(name)
    sp = [SamplingParams(temperature=1.0, top_p=0.95, max_tokens=2000, seed=i) for i in range(len(convs))]
    outs = llm.chat(convs, sp, use_tqdm=False)
    raw = env.path("DATA_DIR") / "templates_raw.json"  # unfiltered output, for audits
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(
        json.dumps([dict(slot=n, text=o.outputs[0].text) for n, o in zip(owners, outs, strict=True)])
    )
    ev = eval_skeletons()
    cands: dict[str, list[str]] = {n: [] for n in names}
    drops: dict[str, dict[str, int]] = {n: {} for n in names}
    for name, o in zip(owners, outs, strict=True):
        for sent in parse(o.outputs[0].text):
            norm = " ".join(sent.lower().split())
            if norm in {" ".join(c.lower().split()) for c in cands[name]}:
                drops[name]["duplicate"] = drops[name].get("duplicate", 0) + 1
                continue
            why = code_filter(name, S[name], sent, ev)
            if why:
                drops[name][why] = drops[name].get(why, 0) + 1
            else:
                cands[name].append(sent)
    # Meaning check by the same model, on the sentence with example values filled in.
    flat = [(n, c) for n in names for c in cands[n]]
    checks = [
        [
            {
                "role": "user",
                "content": CHECK.format(sent=example(S[n], c), meaning=example(S[n], S[n]["meaning"])),
            }
        ]
        for n, c in flat
    ]
    res = llm.chat(checks, SamplingParams(temperature=0.0, max_tokens=4), use_tqdm=False) if checks else []
    path = DIR / "bank.json"
    old = json.loads(path.read_text()) if path.exists() else {"slots": {}, "meta": {}}
    kept: dict[str, list[str]] = {n: [] for n in names}
    for (n, c), r in zip(flat, res, strict=True):
        if r.outputs[0].text.strip().lower().startswith("yes"):
            kept[n].append(c)
        else:
            drops[n]["meaning_check"] = drops[n].get("meaning_check", 0) + 1
    for n in names:
        rng = random.Random(f"bank:{n}")
        ks = kept[n]
        rng.shuffle(ks)
        old["slots"][n] = sorted(ks[: a.keep])
        old["meta"].setdefault("slots", {})[n] = dict(
            kept=len(old["slots"][n]), passed=len(ks), drops=drops[n]
        )
        print(f"{n:28s} kept {len(old['slots'][n]):3d} of {len(ks)} passing; drops {drops[n]}", flush=True)
    old["meta"].update(
        model=pin["repo"], revision=pin["revision"], prompt=PROMPT, check=CHECK, style=STYLE,
        per_call=a.per_call, calls=a.calls, temperature=1.0, top_p=0.95,
        generated=time.strftime("%Y-%m-%d %H:%M"), seconds=round(time.time() - t0),
    )  # fmt: skip
    old["slots"] = dict(sorted(old["slots"].items()))
    path.write_text(json.dumps(old, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
