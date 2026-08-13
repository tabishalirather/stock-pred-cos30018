"""
make_forecast_explainer.py
A four-part pictorial walkthrough of how a forecast is produced, drawn from
the real CBA.AX data wherever possible.

    explainer_1_series.png   the whole price series, and where testing happens
    explainer_2_window.png   one window: 20 days in, h days out
    explainer_3_sliding.png  the window sliding forward, and the overlap
    explainer_4_grid.png     the resulting grid, and what each plot slices

    python make_forecast_explainer.py
"""

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

import regime_lib as rl  # noqa: E402

DARK = "#333333"
GREY = "#9a9a9a"
ORANGE = "#c46a3f"
BLUE = "#2f6f9f"
LIGHT_BLUE = "#dbe6ef"
LIGHT_ORANGE = "#f3e0d4"


# --------------------------------------------------------------- panel 1
def series_diagram():
    frame = rl.load_prices()
    close = frame["Close"].to_numpy()
    dates = frame["Date"].to_numpy()

    figure, axis = plt.subplots(figsize=(9.0, 3.4))
    x = np.arange(len(close))
    axis.plot(x, close, color=DARK, linewidth=1.0)

    inputs, _, _ = rl.build_sequences(frame, horizon=1)
    folds = rl.walk_forward_folds(len(inputs))
    offset = rl.LOOKBACK - 1

    for number, (train_slice, test_slice) in enumerate(folds):
        start = test_slice.start + offset
        stop = test_slice.stop + offset
        axis.axvspan(start, stop, color=ORANGE, alpha=0.10 + 0.03 * number)
        axis.text((start + stop) / 2, 1.02, "fold %d" % (number + 1),
                  fontsize=7.5, ha="center", color="#8a4b28")

    first_test = folds[0][1].start + offset
    axis.axvspan(0, first_test, color=BLUE, alpha=0.08)
    axis.text(first_test / 2, 1.02, "initial training data only",
              fontsize=8, ha="center", color="#1a3d5c")

    ticks = list(range(0, len(close), 120))
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(dates[i])[:7] for i in ticks], fontsize=8)
    axis.set_ylabel("Close price (scaled)")
    axis.set_ylim(0, 1.12)
    axis.set_title("The whole data set, and where the models are tested",
                   fontsize=11)
    axis.grid(alpha=0.25, linewidth=0.4, linestyle="--")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

    figure.tight_layout()
    path = os.path.join(HERE, "explainer_1_series.png")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print("wrote %s" % path)


# --------------------------------------------------------------- panel 2
def window_diagram():
    frame = rl.load_prices()
    close = frame["Close"].to_numpy()

    start = 500
    lookback, horizon = 20, 10
    window = close[start:start + lookback]
    future = close[start + lookback:start + lookback + horizon]

    figure, axis = plt.subplots(figsize=(9.0, 3.6))
    x_in = np.arange(-lookback + 1, 1)
    x_out = np.arange(1, horizon + 1)

    axis.axvspan(-lookback + 0.5, 0.5, color=LIGHT_BLUE, alpha=0.8)
    axis.axvspan(0.5, horizon + 0.5, color=LIGHT_ORANGE, alpha=0.8)

    axis.plot(x_in, window, color=DARK, linewidth=1.6, marker="o",
              markersize=3.5)
    axis.plot(x_out, future, color=DARK, linewidth=1.4, linestyle="--",
              marker="o", markersize=3.5, alpha=0.5)

    for offset in (1, 2, horizon):
        axis.annotate("", xy=(offset, future[offset - 1]),
                      xytext=(0, window[-1]),
                      arrowprops=dict(arrowstyle="->", color=ORANGE,
                                      linewidth=1.0, alpha=0.75,
                                      connectionstyle="arc3,rad=-0.25"))
    axis.text(-lookback / 2, max(window.max(), future.max()) + 0.028,
              "INPUT\n20 trading days the model can see",
              ha="center", fontsize=9.5, color="#1a3d5c")
    axis.text(horizon / 2 + 0.5, max(window.max(), future.max()) + 0.028,
              "OUTPUT\nthe next h days, predicted in one go",
              ha="center", fontsize=9.5, color="#8a4b28")

    for offset, label in ((1, "step 1"), (2, "step 2"), (horizon, "step h")):
        axis.text(offset, future[offset - 1] - 0.030, label, fontsize=8,
                  ha="center", color="#8a4b28")

    axis.set_xlabel("Trading days, relative to the last day the model sees")
    axis.set_ylabel("Close price (scaled)")
    axis.set_title("One forecast: twenty days in, h days out", fontsize=11)
    axis.grid(alpha=0.25, linewidth=0.4, linestyle="--")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

    figure.tight_layout()
    path = os.path.join(HERE, "explainer_2_window.png")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print("wrote %s" % path)


# --------------------------------------------------------------- panel 3
def sliding_diagram():
    frame = rl.load_prices()
    close = frame["Close"].to_numpy()
    start = 500
    lookback, horizon = 20, 6

    figure, axis = plt.subplots(figsize=(9.0, 3.4))

    for row, shift in enumerate([0, 1, 2]):
        y = 2 - row
        base = start + shift
        x_in = np.arange(shift, shift + lookback)
        x_out = np.arange(shift + lookback, shift + lookback + horizon)
        axis.plot(x_in, np.full(lookback, y), color=BLUE, linewidth=7,
                  solid_capstyle="butt", alpha=0.75)
        axis.plot(x_out, np.full(horizon, y), color=ORANGE, linewidth=7,
                  solid_capstyle="butt", alpha=0.75)
        axis.text(-1.5, y, "day %d" % (row + 1), fontsize=8.5, ha="right",
                  va="center", color=GREY)

    axis.annotate("", xy=(1, 2.32), xytext=(0, 2.32),
                  arrowprops=dict(arrowstyle="->", color=DARK, linewidth=1.1))
    axis.text(3.2, 2.34, "each new day, the window slides forward by one",
              fontsize=9, color=DARK, va="center")

    axis.text(9.5, -0.35,
              "19 of the 20 input days are shared between\n"
              "consecutive windows, so neighbouring forecasts\n"
              "are made from almost the same information",
              fontsize=8.8, color=DARK, ha="center", va="top")

    axis.set_xlim(-5, 30)
    axis.set_ylim(-1.15, 2.75)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title("The window slides forward, one day at a time",
                   fontsize=11, loc="left")
    for side in ("top", "right", "bottom", "left"):
        axis.spines[side].set_visible(False)

    figure.tight_layout()
    path = os.path.join(HERE, "explainer_3_sliding.png")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print("wrote %s" % path)


# --------------------------------------------------------------- panel 4
def grid_diagram():
    rows, cols = 7, 8
    figure, axis = plt.subplots(figsize=(9.0, 4.0))

    for r in range(rows):
        for c in range(cols):
            axis.add_patch(Rectangle((c, rows - 1 - r), 0.92, 0.92,
                                     facecolor="#eef2f6", edgecolor="#c8d3dc",
                                     linewidth=0.7))

    # a column = one step ahead, across every window
    for r in range(rows):
        axis.add_patch(Rectangle((1, rows - 1 - r), 0.92, 0.92,
                                 facecolor=ORANGE, edgecolor=DARK,
                                 linewidth=1.0, alpha=0.85))
    # a row = one window's whole forecast path
    for c in range(cols):
        axis.add_patch(Rectangle((c, rows - 1 - 4), 0.92, 0.92,
                                 facecolor=BLUE, edgecolor=DARK,
                                 linewidth=1.0, alpha=0.75))
    # overlap cell
    axis.add_patch(Rectangle((1, rows - 1 - 4), 0.92, 0.92,
                             facecolor="#7a5aa0", edgecolor=DARK,
                             linewidth=1.0))

    axis.text(cols / 2, rows + 0.55, "step ahead   (1, 2, 3, ... h)",
              ha="center", fontsize=10, color=DARK)
    axis.annotate("", xy=(cols - 0.1, rows + 0.25), xytext=(0.1, rows + 0.25),
                  arrowprops=dict(arrowstyle="->", color=DARK, linewidth=1.0))
    axis.text(-0.55, rows / 2, "one row per forecast window\n(one per trading day)",
              rotation=90, va="center", ha="center", fontsize=10, color=DARK)

    axis.text(cols + 0.45, rows - 1 - 0.0 + 0.45,
              "One STEP figure takes a\ncolumn: the same distance\n"
              "ahead, from every window.",
              fontsize=9, color="#8a4b28", va="top")
    axis.text(cols + 0.45, rows - 1 - 4 + 0.45,
              "One TRAJECTORY figure\ntakes a row: a single\n"
              "window's whole forecast.",
              fontsize=9, color="#1a3d5c", va="top")

    axis.set_xlim(-1.4, cols + 4.6)
    axis.set_ylim(-0.6, rows + 1.1)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title("Every forecast, arranged as a grid", fontsize=11, loc="left")
    for side in ("top", "right", "bottom", "left"):
        axis.spines[side].set_visible(False)

    figure.tight_layout()
    path = os.path.join(HERE, "explainer_4_grid.png")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print("wrote %s" % path)


if __name__ == "__main__":
    series_diagram()
    window_diagram()
    sliding_diagram()
    grid_diagram()
