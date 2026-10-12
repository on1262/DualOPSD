"""fig:diagnostic (diagnostic_recurrence) and fig:ablation-dynamics (ablation_dynamics) on Qwen3-4B.

Reads results/mismatch/mismatch.json and results/teacher_diag/teacher_diag.json.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from style import PALETTE, apply_style, load, parse_args, save_figure, style_axes

METHODS = ("opsd", "dualopsd_direct", "dualopsd")
COLORS = {"opsd": PALETTE["gray"], "dualopsd_direct": PALETTE["teal"], "dualopsd": PALETTE["red"]}
MARKERS = {"opsd": "o", "dualopsd_direct": "s", "dualopsd": "^"}
LINESTYLES = {"opsd": (0, (4, 3)), "dualopsd_direct": (0, (1.5, 1.5)), "dualopsd": "-"}
LABELS = {"opsd": "OPSD", "dualopsd_direct": "w/o residual", "dualopsd": "DualOPSD"}
KL = r"$D_{\mathrm{KL}}"


def errorbar(ax, x, mean, sd, method, **kwargs) -> None:
    ax.errorbar(x, mean, yerr=sd, color=COLORS[method], marker=MARKERS[method], linestyle=LINESTYLES[method],
                linewidth=1.9 if method == "dualopsd" else 1.35, markersize=3.5, capsize=1.8, elinewidth=0.8, **kwargs)


def diagnostic_figure(data: dict, output_dir) -> None:
    steps = np.asarray(data["steps"])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))
    for method in METHODS:
        for ax, key in zip(axes[:2], ("teacher_to_student_kl", "student_to_teacher_kl")):
            series = data[key][method]
            errorbar(ax, steps, series["mean"], series["sample_sd"], method,
                     label=LABELS[method] if key == "teacher_to_student_kl" else None)
    for ax, title in zip(axes[:2], (KL + r"(p_T\Vert p_S)$", KL + r"(p_S\Vert p_T)$")):
        ax.set_xticks(steps)
        style_axes(ax, xlabel="Training Step", ylabel="Divergence (nat/token)", title=title)
    x, width = np.arange(2), 0.24
    for offset, method in zip((-width, 0, width), METHODS):
        values = [data["r10"][method][signal] for signal in ("positive_clipped_t2s", "positive_s2t")]
        axes[2].bar(x + offset, [v["mean"] for v in values], width, yerr=[v["sample_sd"] for v in values],
                    color=COLORS[method], edgecolor="black", linewidth=0.7, label=LABELS[method], capsize=1.8,
                    error_kw={"elinewidth": 0.8, "capthick": 0.8}, zorder=3)
    axes[2].set_xticks(x, ["Clipped\n" + KL + r"(p_T\Vert p_S)$", "Positive\n" + KL + r"(p_S\Vert p_T)$"])
    axes[2].set_ylim(0, 0.78)
    style_axes(axes[2], ylabel=r"Top-10 recurrence ($R_{10}$)", title=r"Step 20 $\rightarrow$ 100")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
               handlelength=2.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=0.8)
    save_figure(fig, output_dir, "diagnostic_recurrence")
    plt.close(fig)


def ablation_figure(data: dict, output_dir) -> None:
    steps = np.asarray(data["steps"])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))
    for method in ("dualopsd_direct", "dualopsd"):
        acc = data["accuracy"][method]["steps"]
        for policy, linestyle in (("teacher", "--"), ("student", "-")):
            stats = [acc[str(s)][f"{policy}_accuracy"] for s in steps]
            axes[0].errorbar(steps, [100 * v["mean"] for v in stats], yerr=[100 * v["sample_sd"] for v in stats],
                             color=COLORS[method], marker=MARKERS[method], linestyle=linestyle, linewidth=1.7,
                             markersize=3.5, capsize=1.8, elinewidth=0.8)
        js = data["js_retention"][method]
        errorbar(axes[1], steps, [v["mean"] for v in js], [v["sample_sd"] for v in js], method)
    axes[0].set_ylim(68, 98)
    axes[1].axhline(1.0, color=PALETTE["dark_gray"], linestyle=(0, (4, 3)), linewidth=1.1, zorder=1)
    for position, method in enumerate(METHODS):
        height = 100 * data["clipped_mass"][method]["five_step_mean"]
        axes[2].bar(position, height, width=0.65, color=COLORS[method], edgecolor="black", linewidth=0.7, zorder=3)
        axes[2].text(position, height + 1, f"{height:.1f}", ha="center", va="bottom", fontsize=8)
    axes[2].set_xticks(range(3), ["OPSD", "w/o\nresidual", "DualOPSD"])
    axes[2].set_ylim(45, 70)
    axes[2].set_yticks([45, 50, 55, 60, 65, 70])
    for ax, (title, ylabel) in zip(axes[:2], ((r"Diagnostic-set Accuracy $\uparrow$", "Accuracy (%)"),
                                              (r"Privileged Sensitivity $\uparrow$", "JS retention"))):
        ax.set_xticks(steps)
        style_axes(ax, xlabel="Training Step", ylabel=ylabel, title=title)
    style_axes(axes[2], ylabel="KL mass clipped (%)", title=r"Student KL Clipping $\downarrow$")
    handles = [
        plt.Line2D([], [], color=COLORS[m], marker=MARKERS[m], linestyle=LINESTYLES[m],
                   linewidth=1.8 if m == "dualopsd" else 1.35, markersize=3.5, label=LABELS[m])
        for m in METHODS
    ] + [
        plt.Line2D([], [], color=PALETTE["dark_gray"], linestyle=ls, linewidth=1.2, label=label)
        for ls, label in (("--", "teacher"), ("-", "student"))
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False,
               handlelength=2.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=0.8)
    save_figure(fig, output_dir, "ablation_dynamics")
    plt.close(fig)


def main() -> None:
    args = parse_args(__doc__)
    apply_style()
    diagnostic_figure(load(args.results_dir, "mismatch", "mismatch")["figure"], args.output_dir)
    ablation_figure(load(args.results_dir, "teacher_diag", "teacher_diag"), args.output_dir)


if __name__ == "__main__":
    main()
