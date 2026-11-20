"""Plot the training curves of one run (results/runs/<run>/metrics.jsonl) to results/plots/<run>.png.

SFT runs: train loss (per step, smoothed) and validation loss (per eval, per source), with
the best step marked. GRPO runs: mean training reward per step and validation success
per check, with the best step marked. Small PNGs, committed (results/plots/ is not ignored).

  uv run --extra plots python scripts/plot_run.py <run> [<run> ...]
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INK, MUTED, A, B, C = "#1f2328", "#8c959f", "#0969da", "#cf222e", "#1a7f37"


def smooth(xs: list[float], k: int = 20) -> list[float]:
    out, acc = [], []
    for x in xs:
        acc.append(x)
        acc = acc[-k:]
        out.append(sum(acc) / len(acc))
    return out


def plot(run: str) -> Path:
    rows = [json.loads(line) for line in open(ROOT / "results" / "runs" / run / "metrics.jsonl")]
    summ_p = ROOT / "results" / "runs" / run / "summary.json"
    summ = json.loads(summ_p.read_text()) if summ_p.exists() else {}
    tr = [r for r in rows if r.get("kind") == "train"]
    va = [r for r in rows if r.get("kind") == "val"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=110)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if va and "val_loss" in va[0]:  # SFT
        ax.plot(
            [r["step"] for r in tr],
            smooth([r["loss"] for r in tr]),
            color=MUTED,
            lw=1.2,
            label="train loss (smoothed)",
        )
        ax.plot(
            [r["step"] for r in va],
            [r["val_loss"] for r in va],
            color=A,
            lw=2,
            marker="o",
            ms=3,
            label="validation loss",
        )
        src = sorted({k for r in va for k in r if k.startswith("val_loss_")})
        if len(src) > 1:  # one dashed line per data source
            for key, col in zip(src, [C, B, "#8250df", "#bf8700", "#57606a"], strict=False):
                ax.plot([r["step"] for r in va if key in r], [r[key] for r in va if key in r], color=col, lw=1,
                        ls="--", label=key.replace("val_loss_", "validation loss: "))  # fmt: skip
        ax.set_ylabel("loss per action token")
        best = summ.get("best_step")
        ax.set_yscale("log")
    else:  # GRPO
        ax.plot([r["step"] for r in tr], smooth([r["reward"] for r in tr], 10), color=MUTED, lw=1.2,
                label="train reward (smoothed)")  # fmt: skip
        ax.plot([r["step"] for r in va], [r["val_success"] for r in va], color=A, lw=2, marker="o", ms=3,
                label="validation success")  # fmt: skip
        if va and "val_success_possible" in va[0]:
            ax.plot([r["step"] for r in va], [r["val_honest_failure"] for r in va], color=B, lw=1, ls="--",
                    label="validation honest-failure rate")  # fmt: skip
        ax.set_ylabel("rate")
        ax.set_ylim(0, 1)
        best = summ.get("best_step")
    if best is not None:
        ax.axvline(best, color=INK, lw=0.8, ls=":", label=f"best (step {best})")
    ax.set_xlabel("optimizer step")
    ax.set_title(f"{run}: {summ.get('stop_reason', 'running')}", fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    out = ROOT / "results" / "plots" / f"{run}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for r in sys.argv[1:]:
        print(plot(r))
