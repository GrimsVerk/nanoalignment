import json, sys, urllib.request
def get(u):
    try:
        with urllib.request.urlopen(u, timeout=30) as r: return json.load(r)
    except Exception as e: return {"_err": str(e)}
for mid in sys.argv[1:]:
    m = get(f"https://huggingface.co/api/models/{mid}")
    refs = get(f"https://huggingface.co/api/models/{mid}/refs")
    br = [b["name"] for b in refs.get("branches", [])] if isinstance(refs, dict) else []
    st = (m.get("safetensors") or {}).get("total")
    card = m.get("cardData") or {}
    lic = card.get("license") or [t for t in m.get("tags", []) if t.startswith("license:")]
    ds = card.get("datasets")
    cfg = get(f"https://huggingface.co/{mid}/resolve/main/config.json")
    tc = cfg.get("text_config", cfg) if isinstance(cfg, dict) else {}
    print(f"## {mid} err={m.get('_err')} created={m.get('createdAt','')[:10]} params={st} lic={lic} gated={m.get('gated')} arch={cfg.get('architectures')} vocab={tc.get('vocab_size')} ctx={tc.get('max_position_embeddings')} tie={tc.get('tie_word_embeddings')} layers={tc.get('num_hidden_layers')} hidden={tc.get('hidden_size')}")
    print(f"   datasets={ds} nbranches={len(br)} sample={br[:3]} ... {br[-3:]}")
