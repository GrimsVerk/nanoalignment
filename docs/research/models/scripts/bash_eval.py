"""Few-shot NL-to-bash probe for base models.

Generated commands are NEVER executed. They are only written to a temp file and
checked with `bash -n` (parse only), then matched against reference regexes.

Usage: python bash_eval.py <model_id> [revision] [--label NAME]
Appends one JSON line per model to results/raw.jsonl.
"""
import json, os, re, subprocess, sys, tempfile, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
model_id = args[0]
revision = args[1] if len(args) > 1 else None
label = model_id + (f"@{revision}" if revision else "")

spec = json.load(open(os.path.join(HERE, "prompts.json")))
header = "# Bash one-liners. Each task is followed by a single bash command.\n\n"
shots = "".join(f"# Task: {t}\n$ {c}\n\n" for t, c in spec["fewshot"])


def prompt_for(task):
    return header + shots + f"# Task: {task}\n$"


def bash_syntax_ok(cmd):
    with tempfile.NamedTemporaryFile("w", suffix=".sh", dir="/tmp", delete=False) as f:
        f.write(cmd + "\n")
        p = f.name
    try:
        r = subprocess.run(["bash", "-n", p], capture_output=True, timeout=5)
        return r.returncode == 0
    finally:
        os.unlink(p)


t0 = time.time()
tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModelForCausalLM.from_pretrained(
    model_id, revision=revision, dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
load_s = time.time() - t0
nparams = sum(p.numel() for p in model.parameters())

rows = []
for t in spec["tasks"]:
    prompt = prompt_for(t["task"])
    enc = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id
        )
    gen = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    cmd = gen.split("\n")[0].strip()
    # log-prob of the reference continuation given the prompt
    full = tok(prompt + " " + t["ref"] + "\n", return_tensors="pt").to("cuda")
    n_prompt = enc["input_ids"].shape[1]
    with torch.no_grad():
        logits = model(**full).logits.float()
    lp = torch.log_softmax(logits[0, :-1], -1)
    tgt = full["input_ids"][0, 1:]
    tok_lp = lp.gather(1, tgt[:, None])[:, 0][n_prompt - 1:]
    syntax = bash_syntax_ok(cmd) if cmd else False
    match = any(re.search(p, cmd) for p in t["ok"])
    rows.append({
        "id": t["id"], "cmd": cmd, "raw": gen[:200], "syntax_ok": syntax,
        "auto_match": match, "ref_logprob_sum": tok_lp.sum().item(),
        "ref_logprob_mean": tok_lp.mean().item(), "ref_ntok": len(tok_lp),
    })

# throughput: forward pass, batch 8 x 512, and greedy generation batch 1, 128 tokens
ids = torch.randint(100, 1000, (8, 512), device="cuda")
with torch.no_grad():
    model(input_ids=ids)
    torch.cuda.synchronize(); t1 = time.time()
    for _ in range(3):
        model(input_ids=ids)
    torch.cuda.synchronize()
fwd_tps = 3 * 8 * 512 / (time.time() - t1)
enc = tok(prompt_for("print hello"), return_tensors="pt").to("cuda")
with torch.no_grad():
    torch.cuda.synchronize(); t1 = time.time()
    o = model.generate(**enc, max_new_tokens=128, min_new_tokens=128, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
gen_tps = (o.shape[1] - enc["input_ids"].shape[1]) / (time.time() - t1)
peak_gb = torch.cuda.max_memory_allocated() / 1e9

res = {
    "label": label, "nparams": nparams, "load_s": round(load_s, 1),
    "n_tasks": len(rows), "auto_correct": sum(r["auto_match"] for r in rows),
    "syntax_ok": sum(r["syntax_ok"] for r in rows),
    "mean_ref_logprob_per_tok": sum(r["ref_logprob_mean"] for r in rows) / len(rows),
    "fwd_tok_per_s_b8x512": round(fwd_tps), "gen_tok_per_s_b1": round(gen_tps, 1),
    "peak_vram_gb": round(peak_gb, 2), "rows": rows,
    "torch": torch.__version__,
    "max_rss_gb": round(__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss / 1e6, 2),
}
with open(os.path.join(OUT, "raw.jsonl"), "a") as f:
    f.write(json.dumps(res) + "\n")
print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
for r in rows:
    print(f"{r['id']:>2} syn={int(r['syntax_ok'])} ok={int(r['auto_match'])} lp={r['ref_logprob_mean']:.2f} | {r['cmd']}")
