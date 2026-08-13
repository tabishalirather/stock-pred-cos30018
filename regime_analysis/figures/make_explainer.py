"""
make_explainer.py
Draw the diagram that explains how a forecast is produced and what the two
plotting modes slice out of it. Used as the opening figure of the report.

    python make_explainer.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))

BLUE = "#2f6f9f"
ORANGE = "#c46a3f"
GREY = "#9a9a9a"
DARK = "#333333"


def day_boxes(axis, start_x, count, y, colour, edge=DARK, height=0.42,
              width=0.34, gap=0.06, alpha=1.0):
    """A row of small squares representing consecutive trading days."""
    xs = []
    for i in range(count):
        x = start_x + i * (width + gap)
        axis.add_patch(Rectangle((x, y), width, height, facecolor=colour,
                                 edgecolor=edge, linewidth=0.7, alpha=alpha))
        xs.append(x + width / 2)
    return xs


def main():
    figure, axes = plt.subplots(3, 1, figsize=(9.2, 7.4))

    # ------------------------------------------------ panel A: one forecast
    axis = axes[0]
    axis.set_title("A.  How a single forecast is made", fontsize=11.5,
                   loc="left", fontweight="bold")

    input_xs = day_boxes(axis, 0.4, 8, 1.0, "#dbe6ef")
    output_xs = day_boxes(axis, 0.4 + 8 * 0.40 + 0.5, 4, 1.0, "#f3e0d4")

    axis.text(input_xs[len(input_xs) // 2], 1.62,
              "INPUT: the last 20 trading days\n(prices the model can see)",
              ha="center", fontsize=9, color=DARK)
    axis.text(output_xs[len(output_xs) // 2], 1.62,
              "OUTPUT: the next h days\n(the forecast)",
              ha="center", fontsize=9, color="#8a4b28")

    axis.add_patch(FancyArrowPatch((input_xs[-1] + 0.25, 1.21),
                                   (output_xs[0] - 0.25, 1.21),
                                   arrowstyle="-|>", mutation_scale=13,
                                   linewidth=1.4, color=DARK))
    axis.text((input_xs[-1] + output_xs[0]) / 2, 1.30, "model",
              ha="center", fontsize=9, style="italic", color=DARK)

    for offset, label in zip(range(4), ["+1", "+2", "+3", "+h"]):
        axis.text(output_xs[offset], 0.82, label, ha="center", fontsize=8.5,
                  color="#8a4b28")
    axis.text(output_xs[0], 0.55,
              "each output day is one \"step ahead\"",
              ha="left", fontsize=8.5, color="#8a4b28", style="italic")

    axis.set_xlim(0, 9.0)
    axis.set_ylim(0.4, 2.0)

    # ------------------------------------------------------ panel B: step
    axis = axes[1]
    axis.set_title("B.  Step plots: fix one step, follow it through time",
                   fontsize=11.5, loc="left", fontweight="bold")

    for row, y in enumerate([1.55, 1.0, 0.45]):
        shift = row * 0.46
        day_boxes(axis, 0.4 + shift, 6, y, "#dbe6ef", alpha=0.75)
        out_start = 0.4 + shift + 6 * 0.40 + 0.35
        xs = day_boxes(axis, out_start, 3, y, "#f3e0d4", alpha=0.55)
        # highlight the first forecast step of each window
        axis.add_patch(Rectangle((out_start, y), 0.34, 0.42,
                                 facecolor=ORANGE, edgecolor=DARK,
                                 linewidth=1.1))
        axis.text(xs[0], y + 0.60, "", ha="center")
        axis.text(0.1, y + 0.18, "day %d" % (row + 1), fontsize=8,
                  ha="right", color=GREY)

    axis.text(5.9, 1.15,
              "The highlighted squares are\n"
              "all \"1 step ahead\". Joining\n"
              "them gives the orange line\n"
              "in the step plots.",
              fontsize=9, color=DARK, va="center", ha="left")

    axis.set_xlim(0, 9.0)
    axis.set_ylim(0.25, 2.15)

    # ------------------------------------------------ panel C: trajectory
    axis = axes[2]
    axis.set_title("C.  Trajectory plots: fix one date, show its whole path",
                   fontsize=11.5, loc="left", fontweight="bold")

    day_boxes(axis, 0.4, 6, 1.0, "#dbe6ef")
    out_start = 0.4 + 6 * 0.40 + 0.35
    day_boxes(axis, out_start, 8, 1.0, ORANGE)

    axis.text(0.4 + 3 * 0.40, 1.62, "one chosen starting date",
              ha="center", fontsize=9, color=DARK)
    axis.text(out_start + 4 * 0.40, 1.62,
              "the entire forecast from that date", ha="center",
              fontsize=9, color="#8a4b28")
    axis.text(out_start, 0.62,
              "All of these become the orange line in a trajectory plot,\n"
              "so the horizontal axis is days ahead, not calendar dates.",
              fontsize=9, color=DARK, va="top")

    axis.set_xlim(0, 9.0)
    axis.set_ylim(0.25, 2.0)

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            axis.spines[side].set_visible(False)

    figure.tight_layout(h_pad=1.6)
    output = os.path.join(HERE, "explainer_diagram.png")
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print("wrote %s" % output)


if __name__ == "__main__":
    main()
