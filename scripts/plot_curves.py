"""Plot run.json metrics for one or more runs.

python scripts/plot_curves.py --runs runs/tier_a[:A1] [runs/tier_a_trl[:A2]] --out curves.png
python scripts/plot_curves.py --runs runs/tier_a --copy-run-json viz/dist/run.json
"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = [
    ("reward_mean", "reward_mean (±std)"),
    ("kl", "kl"),
    ("completion_len", "completion_len"),
    ("schema_ok", "schema_ok"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="dir[:label]")
    ap.add_argument("--out", default="curves.png")
    ap.add_argument("--copy-run-json", type=Path, default=None)
    args = ap.parse_args()

    runs = []
    for spec in args.runs:
        path, _, label = spec.partition(":")
        label = label or Path(path).name
        payload = json.loads((Path(path) / "run.json").read_text())
        runs.append((label, payload["steps"]))

    if args.copy_run_json:
        src = Path(args.runs[0].partition(":")[0]) / "run.json"
        args.copy_run_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, args.copy_run_json)
        print(f"copied {src} -> {args.copy_run_json}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (key, title) in zip(axes.flat, METRICS):
        for label, steps in runs:
            x = [s["step"] for s in steps]
            y = [s[key] for s in steps]
            ax.plot(x, y, label=label)
            if key == "reward_mean":
                lo = [s["reward_mean"] - s["reward_std"] for s in steps]
                hi = [s["reward_mean"] + s["reward_std"] for s in steps]
                ax.fill_between(x, lo, hi, alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
