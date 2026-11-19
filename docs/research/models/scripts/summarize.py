"""Build markdown tables from results/raw.jsonl (last entry per label wins)."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = {}
for l in open(os.path.join(ROOT, "results", "raw.jsonl")):
    r = json.loads(l); rows[r["label"]] = r
out = ["| model | params | correct /20 (strict regex) | bash -n ok /20 | mean ref log-prob per token | fwd tok/s (8x512) | gen tok/s (b=1) | peak VRAM GB |",
       "|---|---|---|---|---|---|---|---|"]
for k, r in rows.items():
    out.append(f"| {k} | {r['nparams']/1e6:.0f}M | {r['auto_correct']} | {r['syntax_ok']} | {r['mean_ref_logprob_per_tok']:.2f} | {r['fwd_tok_per_s_b8x512']:,} | {r['gen_tok_per_s_b1']} | {r['peak_vram_gb']} |")
open(os.path.join(ROOT, "results", "bash_table.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
det = []
for k, r in rows.items():
    det.append(f"\n## {k}\n")
    for x in r["rows"]:
        det.append(f"- {x['id']:>2} ok={int(x['auto_match'])} syn={int(x['syntax_ok'])} `{x['cmd'][:120]}`")
open(os.path.join(ROOT, "results", "outputs_by_model.md"), "w").write("\n".join(det) + "\n")
