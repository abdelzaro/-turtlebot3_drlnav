#!/usr/bin/env python3
"""
Draw a bar chart of outcome counts for a 100-episode interval
and add ±2 σ error bars (σ ≈ √count).

Usage:
    python plot_outcome_bar.py <START_EPISODE> <MODEL_NAME> [<MODEL_NAME> ...]
Example:
    python plot_outcome_bar.py 200 ddpg_0
"""

import os, sys, glob, socket
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTCOME_LABELS = [
    "Unknown", "Success", "Collision Wall",
    "Collision Obstacle", "Timeout", "Tumble"
]
INTERVAL = 500          # fixed window length

def main(args=sys.argv[1:]):
    if len(args) < 2:
        print("Usage: python plot_outcome_bar.py <START_EPISODE> <MODEL_NAME> ...")
        return

    start_ep = int(args[0])
    end_ep   = start_ep + INTERVAL
    models   = args[1:]

    fig, ax  = plt.subplots(figsize=(16, 10))
    bar_w    = 0.8 / len(models)
    x_base   = np.arange(len(OUTCOME_LABELS))

    for m_idx, model in enumerate(models):
        # ── locate log ───────────────────────────────────────────────────────
        base_path = os.path.join(
            os.getenv("DRLNAV_BASE_PATH"),
            "src/turtlebot3_drl/model",
            f"{socket.gethostname()}/",
        )
        if "examples" in model:
            base_path = os.path.join(
                os.getenv("DRLNAV_BASE_PATH"), "src/turtlebot3_drl/model/"
            )
        logfile = glob.glob(os.path.join(base_path, model, "_train_*.txt"))
        if len(logfile) != 1:
            print(f"ERROR: found {len(logfile)} log files for {model} in {base_path}")
            continue

        # ── load & slice ─────────────────────────────────────────────────────
        df = pd.read_csv(logfile[0])
        df.columns = [c.strip() for c in df.columns]
        df.rename(columns={"success": "outcome"}, inplace=True)

        window = df[(df["episode"] >= start_ep) & (df["episode"] < end_ep)]
        if window.empty:
            print(f"No data for episodes {start_ep}-{end_ep-1} in {model}")
            continue

        counts = (
            window["outcome"]
            .value_counts()
            .reindex(range(len(OUTCOME_LABELS)))
            .fillna(0)
            .astype(int)
        )

        # ── error bars: ±2√count ────────────────────────────────────────────
        errs = 2 * np.sqrt(counts)

        # ── bar plot ─────────────────────────────────────────────────────────
        x = x_base + m_idx * bar_w
        ax.bar(x, counts, width=bar_w, label=model, yerr=errs, capsize=4)


        # optional console output
        print(f"Model {model}, episodes {start_ep}-{end_ep-1}:")
        for i, label in enumerate(OUTCOME_LABELS):
            print(f"  {label:>18}: {counts[i]}  (±{errs[i]:.1f})")

    # ── styling ─────────────────────────────────────────────────────────────
    ax.set_xticks(x_base + bar_w * (len(models) - 1) / 2)
    ax.set_xticklabels(OUTCOME_LABELS, fontsize=18, rotation=15)
    ax.set_ylabel("Episode count", fontsize=24, fontweight="bold")
    ax.set_xlabel("Outcome",        fontsize=24, fontweight="bold")
    ax.grid(axis="y", linestyle="--")
    ax.legend(fontsize=20)

    # ── save ────────────────────────────────────────────────────────────────
    dt = datetime.now().strftime("%d-%m-%H:%M:%S")
    suffix = "-".join(models).replace(" ", "_").replace("/", "-")
    out_dir = os.path.join(os.getenv("DRLNAV_BASE_PATH"), "util/graphs/")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"outcome_bar_{start_ep}_{end_ep-1}_{dt}__{suffix}.png"
    fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
