"""
make_regime_explainer.py
Draw two explanatory diagrams for the regime report:

  regime_explainer_naive.png   what the naive baseline actually does
  regime_explainer_types.png   real examples of Normal, Volatile and Trending
                               windows, taken from the actual CBA.AX data

    python make_regime_explainer.py
"""

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

import regime_lib as rl  # noqa: E402

DARK = "#333333"
GREY = "#9a9a9a"
ORANGE = "#c46a3f"
BLUE = "#2f6f9f"


# --------------------------------------------------------------------------
def naive_diagram():
    """Show the persistence baseline against a real stretch of prices."""
    frame = rl.load_prices()
    close = frame["Close"].to_numpy()

    start = 640
    lookback = 20
    horizon = 12
    window = close[start:start + lookback]
    future = close[start + lookback:start + lookback + horizon]
    last_price = window[-1]

    figure, axis = plt.subplots(figsize=(8.6, 3.5))

    x_window = np.arange(-lookback + 1, 1)
    x_future = np.arange(1, horizon + 1)

    axis.plot(x_window, window, color=DARK, linewidth=1.5,
              marker="o", markersize=3, label="Prices the model can see")
    axis.plot(x_future, future, color=DARK, linewidth=1.5, linestyle="--",
              marker="o", markersize=3, alpha=0.55,
              label="What actually happened next")
    axis.plot(x_future, np.repeat(last_price, horizon), color=GREY,
              linewidth=2.0, marker="s", markersize=3.4,
              label="Naive forecast: repeat the last price")

    axis.axvline(0.5, color="#bbbbbb", linewidth=0.9, linestyle=":")

    axis.scatter([0], [last_price], s=45, facecolor="white",
                 edgecolor=DARK, zorder=5, linewidth=1.3)
    axis.annotate("today: the last price we know",
                  xy=(0, last_price), xytext=(-13, last_price - 0.055),
                  fontsize=9, color=DARK,
                  arrowprops=dict(arrowstyle="->", color=DARK, linewidth=0.9))
    axis.annotate("the naive forecast is simply\nthis value, held flat",
                  xy=(7, last_price), xytext=(3.4, last_price - 0.062),
                  fontsize=9, color="#666666",
                  arrowprops=dict(arrowstyle="->", color="#888888",
                                  linewidth=0.9))

    axis.set_xlabel("Trading days, relative to today")
    axis.set_ylabel("Close price (scaled)")
    axis.set_title("The naive baseline: assume nothing changes", fontsize=11)
    axis.grid(alpha=0.3, linewidth=0.4, linestyle="--")
    axis.legend(frameon=False, fontsize=8.5, loc="upper left")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

    figure.tight_layout()
    path = os.path.join(HERE, "regime_explainer_naive.png")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print("wrote %s" % path)


# --------------------------------------------------------------------------
def regime_types_diagram():
    """Show one real window of each regime, side by side."""
    frame = rl.load_prices()
    inputs, _, dates = rl.build_sequences(frame, horizon=1)
    volatility, efficiency = rl.window_statistics(inputs)

    train = slice(0, int(0.4 * len(inputs)))
    thresholds = rl.regime_thresholds(volatility[train], efficiency[train])
    labels = rl.assign_regimes(volatility, efficiency, thresholds)

    close_index = rl.FEATURES.index("Close")
    picks = {}
    # for each regime choose a clear example: the most extreme on its own
    # defining statistic, so the picture matches the definition
    for regime, statistic, biggest in (("Normal", volatility, False),
                                       ("Volatile", volatility, True),
                                       ("Trending", efficiency, True)):
        candidates = np.where(labels == regime)[0]
        values = statistic[candidates]
        chosen = candidates[np.argmax(values) if biggest else np.argmin(values)]
        picks[regime] = chosen

    figure, axes = plt.subplots(1, 3, figsize=(9.4, 3.0), sharey=False)
    titles = {
        "Normal": "Normal\nquiet, no strong direction",
        "Volatile": "Volatile\nlarge day-to-day swings",
        "Trending": "Trending\nsteady move in one direction",
    }
    colours = {"Normal": GREY, "Volatile": ORANGE, "Trending": BLUE}

    for axis, regime in zip(axes, ["Normal", "Volatile", "Trending"]):
        row = picks[regime]
        series = inputs[row, :, close_index]
        axis.plot(np.arange(1, len(series) + 1), series,
                  color=colours[regime], linewidth=1.6, marker="o",
                  markersize=2.6)
        axis.set_title(titles[regime], fontsize=9.5)
        axis.set_xlabel("day within the 20-day window", fontsize=8)
        axis.grid(alpha=0.3, linewidth=0.4, linestyle="--")
        axis.tick_params(labelsize=8)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        spread = float(series.max() - series.min())
        axis.text(0.5, -0.42,
                  "%s   |   spans %.2f of the price range"
                  % (str(dates[row])[:10], spread),
                  transform=axis.transAxes, ha="center", fontsize=8,
                  color="#777777")

    axes[0].set_ylabel("Close price (scaled)", fontsize=9)
    figure.suptitle("Real examples of each market condition, from the data",
                    fontsize=11, y=1.02)
    figure.tight_layout()
    path = os.path.join(HERE, "regime_explainer_types.png")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print("wrote %s" % path)


if __name__ == "__main__":
    naive_diagram()
    regime_types_diagram()
