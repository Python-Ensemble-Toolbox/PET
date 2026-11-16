"""The convergence figure: net present value and constraint violation against simulations, on every case."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optimizers import REFERENCE

# Categorical hues in fixed order by method (validated for colour-vision deficiencies on this surface);
# the lighter ones are below 3:1 against it, so every mean line is also labelled at its end.
COLORS = {
    "EnSQP (BFGS)": "#2a78d6",
    "EnOpt + EPF (r = 10)": "#eb6834",
    "EnOpt + EPF (r = 100)": "#1baf7a",
    "EnOpt + EPF (r = 1000)": "#eda100",
    "EnOpt, constraints ignored": "#e87ba4",
}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1a1a19", "#6b6a63", "#e4e3dd"


def held(runs, values, grid):
    """The path as a step function of the simulations spent, held at its last value once the run has stopped."""
    index = np.clip(np.searchsorted(runs, grid, side="right") - 1, 0, len(values) - 1)
    return np.asarray(values, dtype=float)[index]


def convergence(results: dict, path) -> None:
    """Two panels over the same axis of simulations spent: the NPV below the reference, and the largest excess.

    One thin line per case, ending where that run stopped, and a thick line for their mean.
    """
    methods = results["methods"]
    shown = [label for label in COLORS if label in methods and "traces" in methods[label]]
    reference = np.asarray(methods[REFERENCE]["npv"], dtype=float) if REFERENCE in methods else None
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.5, 9.9), sharex=True, facecolor=SURFACE,
                                      gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08})
    limit = max((max(methods[label]["runs"]) for label in shown), default=1)

    ends = []
    for label in shown:
        color, spent = COLORS[label], methods[label]["runs"]
        grid = np.arange(max(spent) + 1)
        gaps, excesses = [], []
        for i, trace in enumerate(methods[label]["traces"]):
            gap = held(trace["runs"], trace["npv"], grid) - (reference[i] if reference is not None else 0.0)
            excess = 100 * held(trace["runs"], trace["violation"], grid)
            gaps.append(gap)
            excesses.append(excess)
            mine = grid <= spent[i]  # the simulations this run spent, rejected trial steps included
            top.plot(grid[mine], gap[mine], color=color, lw=0.7, alpha=0.4)
            bottom.plot(grid[mine], excess[mine], color=color, lw=0.7, alpha=0.4)
        for ax, mean in ((top, np.mean(gaps, axis=0)), (bottom, np.mean(excesses, axis=0))):
            ax.plot(grid, mean, color=color, lw=2.2, solid_capstyle="round", label=label)
            ax.plot(grid[-1], mean[-1], "o", ms=6, color=color, mec=SURFACE, mew=1.5)
        ends.append((np.mean(gaps, axis=0)[-1], grid[-1], label))

    if reference is not None:
        top.axhline(0, color=MUTED, lw=1.2, ls=(0, (4, 3)))
        ends.append((0.0, limit * 1.02, f"reference ({np.mean(methods[REFERENCE]['runs']):.0f} simulations)"))

    # Direct labels at the ends of the mean lines, nudged apart where they would collide
    lo, hi = top.get_ylim()
    gap = 0.045 * (hi - lo)
    placed = []  # (x, y) of the labels so far
    for value, x, label in sorted(ends):
        near = [py for px, py in placed if abs(px - x) < 0.3 * limit]
        y = value if not near or value - max(near) >= gap else max(near) + gap
        placed.append((x, y))
        top.annotate(label, (x, value), xytext=(x + 0.012 * limit, y), color=INK, fontsize=8.5, va="center",
                     annotation_clip=False, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})

    top.set_ylabel("NPV − reference [1e7 USD]" if reference is not None else "net present value [1e7 USD]", color=INK)
    bottom.set_ylabel("largest excess over a limit [%]", color=INK)
    bottom.set_xlabel("simulations", color=INK)
    bottom.axhline(0, color=MUTED, lw=0.8)
    cases = len(methods[shown[0]]["traces"]) if shown else 0
    fig.suptitle(f"What each method reaches for the simulations it spends: {cases} cases (thin) and their mean",
                 x=0.09, y=0.99, ha="left", color=INK, fontsize=11)
    # Most runs end within a few percent of a limit, one method far beyond: linear up to 1 %, logarithmic above
    bottom.set_yscale("symlog", linthresh=1.0, linscale=0.6)
    bottom.set_ylim(-0.15, 400)
    bottom.set_yticks([0, 1, 10, 100], labels=["0", "1", "10", "100"])
    bottom.minorticks_off()
    for ax in (top, bottom):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    top.set_xlim(0, limit * 1.02)
    fig.legend(*top.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.08, 0.97), ncol=3,
               frameon=False, fontsize=8.5, labelcolor=INK, columnspacing=1.6, handlelength=1.6)
    fig.subplots_adjust(right=0.74, left=0.09, top=0.905, bottom=0.06)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
