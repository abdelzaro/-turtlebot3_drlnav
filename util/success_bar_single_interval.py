
#!/usr/bin/env python3
"""
success_bar_single_interval.py

Usage:
    python success_bar_single_interval.py <START_EPISODE> <INTERVAL> <model1> [<model2> ...]

Example:
    python success_bar_single_interval.py 200 100 ddpg_0 td3_0
The script will search for the corresponding training log files in
$DRLNAV_BASE_PATH/src/turtlebot3_drl/model/<HOSTNAME>/<model>/_train_*.txt
(or .../model/<model>/ if 'examples' is in the model name), extract the
success counts between <START_EPISODE> and <START_EPISODE + INTERVAL>,
and plot a bar chart showing the total number of successes in that range.
"""

import os
import glob
import sys
import socket
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main(args=sys.argv[1:]):
    if len(args) < 3:
        print(
            "Usage: python success_bar_single_interval.py <START_EPISODE> <INTERVAL> <model1> [<model2> ...]"
        )
        sys.exit(1)

    START = int(args[0])
    INTERVAL = int(args[1])
    models = args[2:]

    fig, ax = plt.subplots(figsize=(16, 10))

    bar_width = 0.8 / max(len(models), 1)  # distribute bars if multiple models

    for j, model in enumerate(models):
        # Locate logfile
        base_path = os.getenv("DRLNAV_BASE_PATH") + "/src/turtlebot3_drl/model/" + str(socket.gethostname()) + "/"
        if "examples" in model:
            base_path = os.getenv("DRLNAV_BASE_PATH") + "/src/turtlebot3_drl/model/"
        logfile = glob.glob(base_path + model + "/*_train_*.txt")
        if len(logfile) != 1:
            print(f"ERROR: found {len(logfile)} logfiles for: {base_path}{model}")
            continue

        df = pd.read_csv(logfile[0])

        # Use column with potential leading space
        success_col = "success"
        if success_col not in df.columns:
            success_col = " success"  # fallback in case of leading space
        if success_col not in df.columns:
            print(f"ERROR: success column not found in {logfile[0]}")
            continue

        # Check range validity
        if START + INTERVAL > len(df):
            print(f"ERROR: requested range {START} to {START + INTERVAL} exceeds data length ({len(df)})")
            continue

        # Get success count in the specified range
        count = int(df[success_col][START:START + INTERVAL].sum())
        ax.bar(j, count, width=bar_width, label=model)

        print(f"model {model}: {count} successes from episode {START} to {START + INTERVAL}")

    ax.set_xlabel("Model", fontsize=24, fontweight="bold")
    ax.set_ylabel("Number of Successes", fontsize=24, fontweight="bold")
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, fontsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.grid(True, linestyle="--", axis="y")
    ax.legend(fontsize=20)

    dt_string = datetime.now().strftime("%d-%m-%H:%M:%S")
    suffix = "-".join(models).replace(" ", "_").replace("/", "-")
    outfile = os.path.join(
        os.getenv("DRLNAV_BASE_PATH"),
        "util/graphs/",
        f"success_interval_{START}_{START+INTERVAL}_{dt_string}__{suffix}.png",
    )
    plt.savefig(outfile, format="png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
