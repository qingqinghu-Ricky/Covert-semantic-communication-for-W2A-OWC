#!/usr/bin/env python3
"""Plot Fig. 7(a-d) from evaluated CSV files; no hard-coded curves."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "results/fig7")
    parser.add_argument("--output", type=Path, help="Defaults to input/fig7")
    args = parser.parse_args()
    curves = read_rows(args.input / "fig7_abc.csv")
    positions = read_rows(args.input / "fig7_d.csv")
    if any(not row.get("kl_mean") for row in curves):
        raise ValueError(
            "All four panels require KL evaluation; rerun without --skip-kl"
        )
    output = args.output or args.input / "fig7"
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.grid": False,
        }
    )
    fig = plt.figure(figsize=(9.5, 7.2))
    layout = GridSpec(2, 2, figure=fig, wspace=0.34, hspace=0.46)
    a = fig.add_subplot(layout[0, 0])
    split = GridSpecFromSubplotSpec(2, 1, subplot_spec=layout[0, 1], hspace=0.17)
    b_top, b = fig.add_subplot(split[0]), fig.add_subplot(split[1])
    c, d = fig.add_subplot(layout[1, 0]), fig.add_subplot(layout[1, 1])
    styles = {
        "dense": ("W/O sparse mask", "tab:green", "o"),
        "energy_matched": ("W/O sparse mask (energy-matched)", "tab:blue", "+"),
        "sparse": ("W sparse mask", "orangered", "^"),
    }
    for name, (label, color, marker) in styles.items():
        rows = sorted(
            (r for r in curves if r["scheme"] == name),
            key=lambda r: float(r["noise_power_dbm"]),
        )
        if not rows:
            raise ValueError(f"Missing scheme {name}")
        x = [float(r["noise_power_dbm"]) for r in rows]
        kwargs = dict(
            label=label,
            color=color,
            marker=marker,
            markersize=5,
            markerfacecolor="none",
            linewidth=1,
        )
        a.plot(x, [float(r["miou"]) for r in rows], **kwargs)
        c.plot(x, [float(r["kl_mean"]) for r in rows], **kwargs)
        target = b if name == "sparse" else b_top
        target.plot(x, [float(r["sparsity_ratio"]) for r in rows], **kwargs)
    unique = list(dict.fromkeys(r["position"] for r in positions))
    for position, color, marker in zip(
        unique, ["tab:green", "tab:blue", "orangered"], ["o", "^", "+"]
    ):
        rows = sorted(
            (r for r in positions if r["position"] == position),
            key=lambda r: float(r["noise_power_dbm"]),
        )
        d.plot(
            [float(r["noise_power_dbm"]) for r in rows],
            [float(r["kl_mean"]) for r in rows],
            label=position + " m",
            color=color,
            marker=marker,
            markerfacecolor="none",
            linewidth=1,
        )
    a.set_ylabel("Mean IoU")
    a.set_ylim(0.68, 0.72)
    c.set_ylabel("Average KL divergence")
    d.set_ylabel("Average KL divergence")
    b.set_ylabel("Sparsity ratio")
    sparse_ratios = [
        float(row["sparsity_ratio"]) for row in curves if row["scheme"] == "sparse"
    ]
    b.set_ylim(
        min(0.698, min(sparse_ratios) - 0.001),
        max(0.72, max(sparse_ratios) + 0.001),
    )
    b_top.set_ylim(-0.1, 0.1)
    b_top.set_ylabel("Sparsity ratio")
    b_top.tick_params(labelbottom=False)
    for panel in (a, b_top, b, c, d):
        panel.tick_params(top=True, right=True)
        panel.legend(fontsize=8, framealpha=1)
        panel.set_xlim(
            min(float(r["noise_power_dbm"]) for r in curves),
            max(float(r["noise_power_dbm"]) for r in curves),
        )
    for panel, label in zip((a, b, c, d), "abcd"):
        panel.set_xlabel("Noise power (dBm)")
        panel.text(
            0.5,
            -0.55 if panel is b else -0.24,
            f"({label})",
            transform=panel.transAxes,
            horizontalalignment="center",
        )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix("." + suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}.png / .pdf / .svg")


if __name__ == "__main__":
    main()
