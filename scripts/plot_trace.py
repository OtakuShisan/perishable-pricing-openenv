from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot key trends from run trace CSV.")
    parser.add_argument(
        "--input",
        default="artifacts/run_trace.csv",
        help="Path to input run trace CSV.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/trace_plots.png",
        help="Path to output image file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Input trace file not found: {in_path}")

    with in_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Input trace file is empty: {in_path}")

    day = [int(r["day"]) for r in rows]
    profit = [float(r["profit"]) for r in rows]
    reward = [float(r["reward"]) for r in rows]
    waste_total = [
        float(r.get("waste_milk", 0.0)) + float(r.get("waste_banana", 0.0)) + float(r.get("waste_bread", 0.0))
        for r in rows
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(day, profit, marker="o", linewidth=1.5, label="profit")
    axes[0].set_ylabel("Profit")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(day, reward, marker="o", linewidth=1.5, color="tab:green", label="reward")
    axes[1].set_ylabel("Reward")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(day, waste_total, marker="o", linewidth=1.5, color="tab:red", label="waste_total")
    axes[2].set_ylabel("Waste")
    axes[2].set_xlabel("Day")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="best")

    fig.suptitle("Perishable Pricing Trace")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
