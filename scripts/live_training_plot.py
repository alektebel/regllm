#!/usr/bin/env python3
"""Live training curve monitor for GRPO curricular training.

Writes a self-refreshing HTML page with the training plot. Open it
in your browser and it updates every 5 seconds automatically.

Usage:
    python scripts/live_training_plot.py [LOG_FILE]

Then open: file:///tmp/training_live.html
"""

import re
import sys
import glob
import time
import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_HTML = Path("/tmp/training_live.html")
OUT_PNG = Path("/tmp/training_live.png")


def find_log_file() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    pattern = "/tmp/claude-*/**/tasks/*.output"
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        print("No log files found. Pass the path as argument.")
        sys.exit(1)
    latest = max(candidates, key=lambda p: Path(p).stat().st_mtime)
    print(f"Tailing: {latest}")
    return latest


def parse_metrics(text: str) -> list[dict]:
    metrics = []
    for m in re.finditer(
        r"\{'loss': '([^']+)', 'grad_norm': '([^']+)'.*?"
        r"'reward': '([^']+)'.*?'kl': '([^']+)'.*?"
        r"'epoch': '([^']+)'\}",
        text,
    ):
        loss, grad, reward, kl, epoch = m.groups()
        try:
            metrics.append({
                "loss": float(loss),
                "grad_norm": float(grad) if grad != "inf" else 100,
                "reward": float(reward),
                "kl": float(kl),
                "epoch": float(epoch),
            })
        except ValueError:
            pass
    return metrics


def smooth(vals, w=5):
    return [np.mean(vals[max(0, i - w) : i + 1]) for i in range(len(vals))]


def render(metrics: list[dict]) -> None:
    steps = list(range(len(metrics)))
    rewards = [m["reward"] for m in metrics]
    losses = [min(m["loss"], 100) for m in metrics]
    kls = [min(m["kl"], 1e5) for m in metrics]

    # Round boundaries
    boundaries = [0]
    for i in range(1, len(metrics)):
        if metrics[i]["epoch"] <= metrics[i - 1]["epoch"] and i > 1:
            boundaries.append(i)

    recent = rewards[-min(16, len(rewards)) :]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.subplots_adjust(hspace=0.28, top=0.90)

    stats = (
        f"Steps: {len(metrics)}  |  "
        f"Last: {rewards[-1]:.3f}  |  "
        f"Mean(16): {np.mean(recent):.3f}  |  "
        f"Best: {max(rewards):.3f}  |  "
        f"Rounds: {len(boundaries)}"
    )
    fig.suptitle(
        f"GRPO Curricular Training — Qwen3-4B\n{stats}",
        fontsize=12, fontweight="bold",
    )

    # Reward
    ax = axes[0]
    ax.plot(steps, rewards, alpha=0.25, color="steelblue", lw=0.8)
    ax.plot(steps, smooth(rewards, 8), color="steelblue", lw=2.2, label="Reward (w=8)")
    ax.axhline(0.5, color="green", ls="--", alpha=0.5, label="Advance threshold")
    ax.axhline(0, color="gray", ls="-", alpha=0.2)
    ax.set_ylabel("Reward")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Reward", fontsize=10)
    ax.grid(True, alpha=0.15)

    # Loss
    ax = axes[1]
    ax.plot(steps, losses, color="coral", lw=0.8, alpha=0.4)
    ax.plot(steps, smooth(losses, 8), color="coral", lw=2.2)
    ax.set_ylabel("Loss (clipped)")
    ax.set_yscale("log")
    ax.set_title("Training Loss", fontsize=10)
    ax.grid(True, alpha=0.15)

    # KL
    ax = axes[2]
    ax.plot(steps, kls, color="mediumpurple", lw=0.8, alpha=0.4)
    ax.plot(steps, smooth(kls, 8), color="mediumpurple", lw=2.2)
    ax.set_ylabel("KL")
    ax.set_yscale("log")
    ax.set_xlabel("Training Step")
    ax.set_title("KL Divergence", fontsize=10)
    ax.grid(True, alpha=0.15)

    # Stage boundaries
    for ax in axes:
        for b in boundaries:
            ax.axvline(b, color="red", ls=":", alpha=0.3)

    # Save PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()

    # Write self-refreshing HTML
    html = f"""<!DOCTYPE html>
<html><head>
<meta http-equiv="refresh" content="5">
<title>GRPO Training Live</title>
<style>
  body {{ background: #1a1a2e; margin: 0; display: flex;
         justify-content: center; align-items: center; min-height: 100vh; }}
  img {{ max-width: 98vw; max-height: 96vh; }}
</style>
</head><body>
<img src="data:image/png;base64,{img_b64}" />
</body></html>"""
    OUT_HTML.write_text(html)


def main():
    log_file = find_log_file()
    print(f"Open in browser: file://{OUT_HTML}")
    print("Updating every 5s. Ctrl+C to stop.\n")

    prev_len = 0
    while True:
        try:
            text = Path(log_file).read_text(errors="replace")
        except FileNotFoundError:
            time.sleep(2)
            continue

        metrics = parse_metrics(text)
        if not metrics:
            time.sleep(3)
            continue

        if len(metrics) != prev_len:
            prev_len = len(metrics)
            render(metrics)
            print(
                f"\r  step {len(metrics):>4d}  "
                f"reward={metrics[-1]['reward']:+.3f}  "
                f"loss={metrics[-1]['loss']:.4f}  "
                f"kl={metrics[-1]['kl']:.1f}",
                end="", flush=True,
            )

        time.sleep(5)


if __name__ == "__main__":
    main()
