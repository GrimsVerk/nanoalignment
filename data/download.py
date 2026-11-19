"""Stream documents from pinned dataset files into DATA_DIR/raw/<source>.jsonl.

Streaming reads only what is needed, so a smoke test does not download whole shards.
Each line: {"text": ...}. Chat data is flattened to "User: ... / Assistant: ..." text.

  uv run python -m data.download --source fineweb_edu --docs 4000
  uv run python -m data.download --config data/smoke.yaml     # every source in download_docs
"""

import argparse
import json

from common import config, env


def _flatten_chat(msgs: list[dict]) -> str:
    names = {"user": "User", "assistant": "Assistant", "system": "System"}
    return "\n\n".join(f"{names.get(m['role'], m['role'])}: {m['content']}" for m in msgs)


def stream(source: str, n_docs: int):
    from datasets import load_dataset

    ref = config.dataset_ref(source)
    files = [f"hf://datasets/{ref['repo']}@{ref['revision']}/{f}" for f in ref["files"]]
    fmt = "parquet" if files[0].endswith(".parquet") else "json"
    ds = load_dataset(fmt, data_files=files, split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= n_docs:
            break
        if "messages_field" in ref:
            yield _flatten_chat(row[ref["messages_field"]])
        else:
            yield row[ref["text_field"]]


def download(source: str, n_docs: int) -> dict:
    out = env.path("DATA_DIR") / "raw" / f"{source}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = chars = 0
    with open(out, "w") as f:
        for text in stream(source, n_docs):
            f.write(json.dumps({"text": text}) + "\n")
            n += 1
            chars += len(text)
    ref = config.dataset_ref(source)
    meta = dict(
        source=source, repo=ref["repo"], revision=ref["revision"], files=ref["files"], docs=n, chars=chars
    )
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source")
    ap.add_argument("--docs", type=int, default=1000)
    ap.add_argument("--config")
    a = ap.parse_args()
    if a.config:
        for src, n in config.load(a.config)["download_docs"].items():
            download(src, n)
    else:
        download(a.source, a.docs)


if __name__ == "__main__":
    main()
