"""
make_regime_figures.py
Generate one Figure-4-style plot for every cell of Table 7: each forecast
horizon (1, 2, 100, 200 days) crossed with each market regime (Normal,
Volatile, Trending). Twelve figures in total.

    python make_regime_figures.py

Each figure shows the actual price, the LSTM forecast, the KAN forecast and
the naive persistence baseline, restricted to the test windows that fall in
one regime.

A NOTE ON THE HORIZONTAL AXIS
-----------------------------
Filtering to a single regime leaves a set of dates that are not consecutive:
a Volatile stretch in March and another in June, with quiet weeks between
that belong to a different regime. Drawing a continuous line across those
gaps would imply a continuity that does not exist.

So the horizontal axis counts the windows that belong to the regime, in
chronological order, rather than running along a calendar. Tick labels give
the real date at that position, and a gap between consecutive ticks of more
than a few days means the intervening period was a different regime. Points
are drawn as markers joined by thin lines to keep this visible.

All five walk-forward folds are pooled so that each regime has enough windows
to be worth plotting, using seed 0 throughout.
"""

import glob
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

import driver  # noqa: E402

def _results_dir(base):
    """Honour the REGIME_RESULTS_DIR override used by driver.py, so runs
    written to an independent directory can be plotted the same way."""
    path = os.environ.get("REGIME_RESULTS_DIR") or os.path.join(base, "results")
    if not os.path.isabs(path):
        path = os.path.join(base, path)
    return path


PRED = os.path.join(_results_dir(PARENT), "preds")
PRED_PYKAN = os.path.join(_results_dir(PARENT), "preds_pykan")
OUT_DIR = HERE

HORIZONS = [1, 2, 100, 200]
REGIMES = ["Normal", "Volatile", "Trending"]
SEED = 0
STEP = 1                      # plot the 1-step-ahead component

COLOUR_ACTUAL = "#333333"
COLOUR_LSTM = "#2f6f9f"
COLOUR_KAN = "#c46a3f"
COLOUR_NAIVE = "#9a9a9a"


def _fold(path):
    return int(os.path.basename(path).split("__f")[-1].split(".")[0])


def load_all_folds(model, horizon, seed=SEED):
    """Pool every fold for one model at one horizon, in chronological order."""
    if model == "kan":
        pattern = os.path.join(PRED_PYKAN, "kan__-__h%d__s%d__f*.npz" % (horizon, seed))
    elif model == "naive":
        pattern = os.path.join(PRED, "naive__-__h%d__s0__f*.npz" % horizon)
    else:
        config = driver.PRIMARY_LSTM_CONFIG[horizon]
        pattern = os.path.join(PRED, "lstm__%s__h%d__s%d__f*.npz"
                               % (config, horizon, seed))

    paths = sorted(glob.glob(pattern), key=_fold)
    if not paths:
        return None

    predictions, truths, labels, dates = [], [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            predictions.append(data["prediction"].astype(float))
            truths.append(data["truth"].astype(float))
            labels.append(data["labels"])
            dates.append(data["dates"])
    return (np.concatenate(predictions), np.concatenate(truths),
            np.concatenate(labels), np.concatenate(dates))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def build_one(horizon, regime):
    lstm = load_all_folds("lstm", horizon)
    kan = load_all_folds("kan", horizon)
    naive = load_all_folds("naive", horizon)
    if lstm is None or kan is None:
        print("missing runs for horizon %d" % horizon)
        return None

    labels = lstm[2]
    mask = labels == regime
    count = int(mask.sum())
    if count == 0:
        print("no %s windows at horizon %d" % (regime, horizon))
        return None

    index = STEP - 1
    truth = lstm[1][mask][:, index]
    dates = lstm[3][mask]
    series = {
        "LSTM": lstm[0][mask][:, index],
        "KAN": kan[0][mask][:, index],
    }
    if naive is not None:
        series["Naive"] = naive[0][mask][:, index]

    # ------------------------------------------------------------ plotting
    figure, axis = plt.subplots(figsize=(8.6, 3.3))
    x = np.arange(count)

    axis.plot(x, truth, color=COLOUR_ACTUAL, linewidth=1.2, marker="o",
              markersize=2.4, label="Actual", zorder=3)
    styles = {"LSTM": COLOUR_LSTM, "KAN": COLOUR_KAN, "Naive": COLOUR_NAIVE}
    for name, values in series.items():
        axis.plot(x, values, color=styles[name], linewidth=0.85, alpha=0.9,
                  marker="o", markersize=1.9, label=name)

    every = max(1, count // 6)
    positions = list(range(0, count, every))
    axis.set_xticks(positions)
    axis.set_xticklabels([str(dates[i])[:7] for i in positions], fontsize=8)
    axis.set_ylabel("Close price (scaled)")
    axis.set_xlabel("%s windows in chronological order (%d of them)"
                    % (regime, count), fontsize=9)
    axis.set_title("%d-day horizon, %s conditions" % (horizon, regime),
                   fontsize=10)
    axis.grid(alpha=0.3, linewidth=0.4, linestyle="--")
    axis.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

    figure.tight_layout()
    stem = "regime_h%d_%s" % (horizon, regime.lower())
    path = os.path.join(OUT_DIR, stem + ".png")
    figure.savefig(path, dpi=150)
    plt.close(figure)

    stats = {name: rmse(values, truth) for name, values in series.items()}
    return {"stem": stem, "horizon": horizon, "regime": regime,
            "count": count, "rmse": stats,
            "first": str(dates[0]), "last": str(dates[-1])}


def main():
    summary = []
    for horizon in HORIZONS:
        for regime in REGIMES:
            result = build_one(horizon, regime)
            if result:
                summary.append(result)
                print("%-4d %-9s n=%-4d LSTM %.4f  KAN %.4f  Naive %.4f"
                      % (horizon, regime, result["count"],
                         result["rmse"]["LSTM"], result["rmse"]["KAN"],
                         result["rmse"].get("Naive", float("nan"))))

    with open(os.path.join(OUT_DIR, "regime_figure_stats.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print("\nwrote %d figures and regime_figure_stats.json" % len(summary))


if __name__ == "__main__":
    main()
