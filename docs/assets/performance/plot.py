"""Render README figures with Matplotlib from the PR tables in data.json.

Run: python docs/assets/performance/plot.py
PR30 P8r/P9 are separate split=1 reruns. PR31 bars use reported absolute
values; change labels preserve the percentages published in the PR.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, PercentFormatter, StrMethodFormatter

plt.switch_backend("Agg")
ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "data.json").read_text())
INK = "#172b4d"
MUTED = "#60718a"
COLORS = ["#a5afbd", "#2563eb", "#0d9488", "#d99732"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": INK,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#dce3ec",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e8edf3", linewidth=0.8)
    ax.tick_params(axis="both", length=0, pad=9)


def runtime():
    fig, ax = plt.subplots(figsize=(15, 7.6))
    fig.subplots_adjust(left=0.08, right=0.965, top=0.72, bottom=0.24)
    fig.text(
        0.08, 0.935, "Feature stacking across request concurrency", fontsize=24, fontweight="bold"
    )
    fig.text(
        0.08,
        0.887,
        "Ouro-1.4B BF16 · B300 · 1,024 input / 128 output tokens · end-to-end throughput",
        fontsize=12,
        color=MUTED,
    )
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8.4, 9.4])
    labels = [
        "FA4",
        "+ Early\nexit",
        "Delayed\nexit",
        "+ Async",
        "+ Multi-\nstream",
        "+ Static\nbuffers",
        "+ Padding",
        "+ CUDA\nGraphs",
        "Graphs\nsplit=1",
        "+ Resident\nstate",
    ]
    ax.axvspan(7.85, 9.85, color="#f0f4f9")
    selected = [r for r in DATA["runtime_e2e"] if r["context"] == "1K"]
    for row, color, marker in zip(selected, ["#2563eb", "#0d9488", "#ac55cb"], ["o", "s", "^"]):
        values = np.array(row["throughput"])
        ratios = values / values[0]
        label = f"Concurrency {row['concurrency']}  ·  FA4 baseline {values[0]:,.2f} tok/s"
        ax.plot(
            x[:8], ratios[:8], color=color, marker=marker, linewidth=2.8, markersize=7, label=label
        )
        ax.plot(x[8:], ratios[8:], color=color, marker=marker, linewidth=2.8, markersize=7)
    ax.axhline(1, color="#a5afbd", linewidth=1, linestyle=(0, (4, 4)))
    ax.set_ylim(0, 3.35)
    ax.set_xlim(-0.3, 9.85)
    ax.set_ylabel("Throughput / each case's FA4 baseline", labelpad=13)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}×"))
    ax.set_xticks(x, labels, fontsize=11)
    ax.text(8.85, 3.17, "Separate split=1 rerun", ha="center", fontsize=10, color=MUTED)
    ax.annotate(
        "2.97×",
        xy=(7, selected[0]["throughput"][7] / selected[0]["throughput"][0]),
        xytext=(0, 13),
        textcoords="offset points",
        ha="center",
        fontsize=15,
        color="#2563eb",
        fontweight="bold",
    )
    style(ax)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper left",
        bbox_to_anchor=(0.073, 0.855),
        frameon=False,
        fontsize=11,
        ncol=1,
        labelspacing=0.6,
    )
    fig.text(
        0.08,
        0.11,
        "Source: PR #30 · Median of 3 trials; 2×concurrency requests. High"
        "er is better. Triton excluded.",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.08,
        0.074,
        "Delayed exit changes the policy. Shaded points are cross-campaign"
        " comparisons to FA4; only their pair is matched.",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.08,
        0.038,
        "Timing excludes HTTP, tokenization, loading and warmup. Some toke"
        "n/exit-depth differences remain unresolved.",
        fontsize=10,
        color=MUTED,
    )
    fig.savefig(ROOT / "pr30-feature-stacking.png", dpi=180)
    plt.close(fig)


def serving(rate):
    rows = {
        r["configuration"]: np.array(r["metrics"]) for r in DATA["pd_serving"] if r["rate"] == rate
    }
    changes = {
        r["configuration"]: r["reported_change_pct"]
        for r in DATA["pd_serving"]
        if r["rate"] == rate
    }
    baseline = rows["4 replicas"]
    fig, ax = plt.subplots(figsize=(15, 7.8))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.68, bottom=0.23)
    fig.text(
        0.075,
        0.935,
        f"Prefill/decode disaggregation · {rate} requests/s",
        fontsize=25,
        fontweight="bold",
    )
    highlight = 100 + np.array(changes["2P2D"])
    fig.text(
        0.075,
        0.875,
        f"2P2D retains {highlight[0]:.1f}% throughput  |  "
        f"TTFT −{100 - highlight[1]:.1f}%  |  ITL −{100 - highlight[3]:.1f}%",
        fontsize=17,
        color="#0d9488",
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.825,
        "Four GPUs per configuration · Ouro-1.4B BF16 · 512 ShareGPT promp"
        "ts / phase · 128 output tokens",
        fontsize=11,
        color=MUTED,
    )
    x = np.arange(5) * 1.5
    for i, (name, color) in enumerate(zip(["4 replicas", "1P3D", "2P2D", "3P1D"], COLORS)):
        ratio = rows[name] / baseline * 100
        bars = ax.bar(
            x + (i - 1.5) * 0.26,
            ratio,
            width=0.225,
            color=color,
            label="4 replicas (baseline)" if i == 0 else name,
            zorder=3,
        )
        for bar, change in zip(bars, changes[name]):
            label = "100%" if i == 0 else f"{change:+.1f}%"
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold" if i == 2 else "normal",
            )
    ax.axhline(100, color="#7d8ca0", linewidth=1.1, linestyle=(0, (4, 4)), zorder=2)
    labels = ["Throughput ↑", "TTFT P95 ↓", "TPOT P95 ↓", "ITL P99 ↓", "E2E P95 ↓"]
    units = [
        f"{baseline[0]:,.1f} tok/s",
        f"{baseline[1]:.3f} s",
        f"{baseline[2]:.1f} ms",
        f"{baseline[3]:.1f} ms",
        f"{baseline[4]:.2f} s",
    ]
    ax.set_xticks(
        x, [f"{label}\nBaseline: {unit}" for label, unit in zip(labels, units)], fontsize=11
    )
    ax.set_ylim(0, 165 if rate == 4 else 450)
    ax.yaxis.set_major_locator(MultipleLocator(25 if rate == 4 else 100))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel("Relative to four replicas (100%)", labelpad=12)
    style(ax)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper left",
        bbox_to_anchor=(0.068, 0.795),
        ncol=4,
        frameon=False,
        fontsize=12,
        columnspacing=2.0,
    )
    fig.text(
        0.075,
        0.125,
        "Labels show change vs baseline. Throughput: higher is better. All"
        " latency metrics: lower is better.",
        fontsize=11,
        color=MUTED,
    )
    fig.text(
        0.075,
        0.084,
        "Source: PR #31 · Equal-weight initial/replay averages; latency va"
        "lues average phase percentiles. P = Prefill; D = Decode.",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.075,
        0.043,
        "Fixed offered load, not peak capacity. Mixed isolated/concurrent-"
        "host measurements. Change labels use PR-reported percentages.",
        fontsize=10,
        color=MUTED,
    )
    fig.savefig(ROOT / f"pr31-pd-serving-{rate}rps.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    runtime()
    serving(4)
    serving(8)
