"""Matplotlib theme shared by the paper figures (compact serif style, fixed palette)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = {
    "red": "#C00000",
    "teal": "#00AEBB",
    "gray": "#8A8B8C",
    "light_gray": "#D9D9D9",
    "dark_gray": "#3A3A3A",
}


def apply_style(font_scale: float = 1.0) -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5 * font_scale,
        "axes.labelsize": 9.0 * font_scale,
        "axes.labelweight": "bold",
        "axes.titlesize": 9.5 * font_scale,
        "axes.titleweight": "bold",
        "xtick.labelsize": 7.5 * font_scale,
        "ytick.labelsize": 7.5 * font_scale,
        "legend.fontsize": 7.5 * font_scale,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "grid.color": PALETTE["light_gray"],
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def style_axes(ax: plt.Axes, *, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
               title: Optional[str] = None, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="both", length=0, pad=2)
    ax.grid(axis=grid_axis, zorder=0)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=2)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=3)
    if title:
        ax.set_title(title, pad=4)


def save_figure(fig: plt.Figure, output_dir: Path, name: str, dpi: int = 300) -> None:
    """Write ``<name>.pdf`` and a ``<name>.png`` preview."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{name}.{suffix}", bbox_inches="tight", dpi=dpi, pad_inches=0.03)
    print(output_dir / f"{name}.pdf")


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="analysis outputs (default: $DUALOPSD_SCRATCH/results)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="figure directory (default: <results-dir>/figures)")
    args = parser.parse_args()
    if args.results_dir is None:
        if not os.environ.get("DUALOPSD_SCRATCH"):
            parser.error("set DUALOPSD_SCRATCH or pass --results-dir")
        args.results_dir = Path(os.environ["DUALOPSD_SCRATCH"]) / "results"
    if args.output_dir is None:
        args.output_dir = args.results_dir / "figures"
    return args


def load(results_dir: Path, exp: str, name: str) -> dict[str, Any]:
    return json.loads((results_dir / exp / f"{name}.json").read_text(encoding="utf-8"))
