from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def to_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    val = row.get(key)
    if val is None or val == "":
        return default
    return float(val)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-step trends from artifacts/run_trace.csv")
    parser.add_argument("--input", default="artifacts/run_trace.csv", help="Path to run_trace.csv")
    parser.add_argument("--output", default="artifacts/trace_plots.png", help="Where to save the plot PNG")
    parser.add_argument("--show", action="store_true", help="Show the plot window (if supported)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing trace CSV: {input_path}")

    rewards: List[float] = []
    profits: List[float] = []
    waste_totals: List[float] = []
    days: List[int] = []

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            days.append(int(row.get("day", 0)))
            rewards.append(to_float(row, "reward"))
            profits.append(to_float(row, "profit"))
            waste_total = (
                to_float(row, "waste_milk")
                + to_float(row, "waste_banana")
                + to_float(row, "waste_bread")
            )
            waste_totals.append(waste_total)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(days, rewards, linewidth=2)
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(days, profits, linewidth=2, color="tab:green")
    axes[1].set_ylabel("Profit")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(days, waste_totals, linewidth=2, color="tab:red")
    axes[2].set_ylabel("Waste (units)")
    axes[2].set_xlabel("Day index within trace")
    axes[2].grid(True, alpha=0.25)

    fig.suptitle("Per-Step Trends: Reward / Profit / Waste", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to: {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

