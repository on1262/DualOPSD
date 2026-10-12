"""fig:beta-validation: held-out accuracy versus residual weight (from results/beta/beta.json)."""

from __future__ import annotations

import matplotlib.pyplot as plt

from style import PALETTE, apply_style, load, parse_args, save_figure, style_axes

NAME = "heldout_beta_accuracy"
PANELS = (("qwen3-1.7b", "Qwen3-1.7B"), ("qwen3-4b", "Qwen3-4B"), ("qwen3-8b", "Qwen3-8B"))


def main() -> None:
    args = parse_args(__doc__)
    data = load(args.results_dir, "beta", "beta")["figure"]
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.05), sharex=True)
    for index, (axis, (model, title)) in enumerate(zip(axes, PANELS)):
        betas, accuracy, selected = data[model]["betas"], data[model]["accuracy"], data[model]["selected_beta"]
        axis.plot(betas, accuracy, color=PALETTE["red"], linestyle="-", marker="o", markersize=4.2,
                  markeredgecolor="white", markeredgewidth=0.55, linewidth=2.0, label="Candidates", zorder=3)
        axis.scatter([selected], [accuracy[betas.index(selected)]], color=PALETTE["red"], edgecolor="white",
                     linewidth=0.7, marker="*", s=95, label=r"Selected $\beta$", zorder=4)
        style_axes(axis, xlabel=r"Residual Weight $\beta$", ylabel="Held-out Accuracy (%)" if index == 0 else None,
                   title=f"({chr(ord('a') + index)}) {title}")
        axis.set_xticks([0.1, 0.5, 0.8, 1.1])
        axis.set_xlim(0.05, 1.15)
        axis.margins(y=0.15)
        axis.legend(loc="best", frameon=False, handlelength=1.8, handletextpad=0.35, borderaxespad=0.35,
                    labelspacing=0.25)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.22, top=0.86, wspace=0.16)
    save_figure(fig, args.output_dir, NAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
