"""
make_decay_figures.py
Show how forecast quality decays as the models are asked to look further
ahead. For a given horizon, the same set of forecasts is plotted at several
different steps ahead: step 1, then progressively further out.

Produces eight figures, four for the 100-day horizon and four for the
200-day horizon, plus a summary curve of error against step.

    python make_decay_figures.py
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

PRED = os.path.join(PARENT, "results", "preds")
PRED_PYKAN = os.path.join(PARENT, "results", "preds_pykan")

SEED = 0
PLAN = {100: [1, 25, 50, 100], 200: [1, 50, 100, 200]}

COLOUR_ACTUAL = "#333333"
COLOUR_LSTM = "#2f6f9f"
COLOUR_KAN = "#c46a3f"
COLOUR_NAIVE = "#9a9a9a"


def _fold(path):
    return int(os.path.basename(path).split("__f")[-1].split(".")[0])


def load(model, horizon, seed=SEED):
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
    predictions, truths, dates = [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            predictions.append(data["prediction"].astype(float))
            truths.append(data["truth"].astype(float))
            dates.append(data["dates"])
    return (np.concatenate(predictions), np.concatenate(truths),
            np.concatenate(dates))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def one_figure(horizon, step, runs):
    lstm, kan, naive = runs
    index = step - 1
    truth = lstm[1][:, index]
    dates = lstm[3] if len(lstm) > 3 else lstm[2]

    series = {"LSTM": lstm[0][:, index], "KAN": kan[0][:, index]}
    if naive is not None:
        series["Naive"] = naive[0][:, index]

    figure, axis = plt.subplots(figsize=(8.6, 3.2))
    x = np.arange(len(truth))
    axis.plot(x, truth, color=COLOUR_ACTUAL, linewidth=1.2, label="Actual",
              zorder=3)
    colours = {"LSTM": COLOUR_LSTM, "KAN": COLOUR_KAN, "Naive": COLOUR_NAIVE}
    for name, values in series.items():
        axis.plot(x, values, color=colours[name], linewidth=0.85, alpha=0.9,
                  label=name)

    every = max(1, len(truth) // 6)
    positions = list(range(0, len(truth), every))
    axis.set_xticks(positions)
    axis.set_xticklabels([str(dates[i])[:7] for i in positions], fontsize=8)
    axis.set_ylabel("Close price (scaled)")
    axis.set_title("%d-day model, predicting %d day%s ahead"
                   % (horizon, step, "" if step == 1 else "s"), fontsize=10)
    axis.grid(alpha=0.3, linewidth=0.4, linestyle="--")
    axis.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

    figure.tight_layout()
    stem = "decay_h%d_step%d" % (horizon, step)
    figure.savefig(os.path.join(HERE, stem + ".png"), dpi=150)
    plt.close(figure)

    return {"stem": stem, "horizon": horizon, "step": step,
            "n": int(len(truth)),
            "rmse": {name: rmse(values, truth) for name, values in series.items()}}


def curve_figure(all_stats):
    """Error against step, for both horizons, on one chart."""
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    for axis, horizon in zip(axes, sorted(PLAN)):
        rows = [r for r in all_stats if r["horizon"] == horizon]
        rows.sort(key=lambda r: r["step"])
        steps = [r["step"] for r in rows]
        for name, colour in (("LSTM", COLOUR_LSTM), ("KAN", COLOUR_KAN),
                             ("Naive", COLOUR_NAIVE)):
            values = [r["rmse"].get(name) for r in rows]
            axis.plot(steps, values, marker="o", markersize=4, linewidth=1.4,
                      color=colour, label=name)
        axis.set_title("%d-day model" % horizon, fontsize=10)
        axis.set_xlabel("days ahead being predicted")
        axis.grid(alpha=0.3, linewidth=0.4, linestyle="--")
        axis.tick_params(labelsize=8)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    axes[0].set_ylabel("RMSE (scaled)")
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("How error grows with distance ahead", fontsize=11)
    figure.tight_layout()
    path = os.path.join(HERE, "decay_curve.png")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print("wrote %s" % path)


def main():
    stats = []
    for horizon, steps in PLAN.items():
        lstm = load("lstm", horizon)
        kan = load("kan", horizon)
        naive = load("naive", horizon)
        if lstm is None or kan is None:
            print("missing runs for horizon %d" % horizon)
            continue
        runs = (lstm + (lstm[2],), kan, naive)
        for step in steps:
            result = one_figure(horizon, step, runs)
            stats.append(result)
            print("h=%-4d step=%-4d n=%-4d LSTM %.4f  KAN %.4f  Naive %.4f"
                  % (horizon, step, result["n"], result["rmse"]["LSTM"],
                     result["rmse"]["KAN"], result["rmse"].get("Naive", float("nan"))))

    curve_figure(stats)
    with open(os.path.join(HERE, "decay_stats.json"), "w") as handle:
        json.dump(stats, handle, indent=2)
    print("\nwrote %d decay figures + curve" % len(stats))


if __name__ == "__main__":
    main()
