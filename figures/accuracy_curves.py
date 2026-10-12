"""fig:acc-curves: Qwen3-4B three-benchmark mean avg@12 over training steps (from results/curves/curves.json)."""

from __future__ import annotations

import statistics

import matplotlib.pyplot as plt

from style import PALETTE, apply_style, load, parse_args, save_figure, style_axes

NAME = "accuracy_curves"  # fig:acc-curves
BENCHMARKS = ("aime24", "aime25", "hmmt25")
# Same colors, markers, and line styles as fig:diagnostic.
METHODS = (
    (("opsd",), "OPSD", PALETTE["gray"], (0, (4, 3)), "o", 1.35),
    (("dualopsd",), "DualOPSD", PALETTE["red"], "-", "^", 1.9),
)


def macro_curve(raw: dict) -> tuple[list[int], list[float], list[float]]:
    """Per seed, average avg@12 over the benchmarks; then mean and sample SD across seeds."""
    steps = sorted(raw, key=int)
    means, spreads = [], []
    for step in steps:
        per_seed = [100.0 * statistics.mean(seed["datasets"][b]["avg_at_12"] for b in BENCHMARKS)
                    for seed in raw[step].values()]
        means.append(statistics.mean(per_seed))
        spreads.append(statistics.stdev(per_seed))
    return [int(s) for s in steps], means, spreads


def main() -> None:
    args = parse_args(__doc__)
    raw = load(args.results_dir, "curves", "curves")["raw"]
    apply_style(font_scale=1.25)
    fig, axis = plt.subplots(figsize=(2.64, 1.68))
    for keys, label, color, linestyle, marker, linewidth in METHODS:
        steps, accuracy, spread = macro_curve(next(raw[k] for k in keys if k in raw))
        axis.errorbar(steps, accuracy, yerr=spread, label=label, color=color, linestyle=linestyle, marker=marker,
                      linewidth=linewidth, markersize=3.5, capsize=1.8, elinewidth=0.8, zorder=3)
    style_axes(axis, xlabel="Training Step", ylabel="Mean avg@12 (%)")
    for spine in ("left", "bottom"):
        axis.spines[spine].set_linewidth(1.0)
    axis.set_xticks([20, 40, 60, 80, 100])
    axis.set_xlim(15, 105)
    axis.set_yticks([20, 25, 30, 35])
    axis.set_ylim(17, 38.5)
    axis.legend(loc="upper left", frameon=False, handlelength=2.2, handletextpad=0.4, borderaxespad=0.2,
                labelspacing=0.25)
    fig.tight_layout(pad=0.2)
    save_figure(fig, args.output_dir, NAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
