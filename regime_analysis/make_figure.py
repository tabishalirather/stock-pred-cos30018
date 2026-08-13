"""
make_figure.py
Build Figure-4-style forecast plots for any horizon from the logged experiment
runs. Change the settings in the CONFIG block below and run the file. Nothing
else needs editing.

    python make_figure.py

Outputs a PNG for viewing and a .tex fragment (pgfplots) that can be pasted
straight into the paper, matching the style of the other figures.

--------------------------------------------------------------------------
TWO WAYS TO PLOT A MULTI-STEP FORECAST
--------------------------------------------------------------------------
At the 1-day horizon each test window produces one number, so a time-series
plot is unambiguous. At 100 or 200 days each window produces a whole path,
and there are two sensible pictures. Pick with MODE:

  MODE = "step"
      Fix one step ahead (STEP) and plot it across every test date.
      "How well does the model predict STEP days ahead, over time?"
      This is the direct generalisation of Figure 4 and is usually what you
      want for comparing models.

  MODE = "trajectory"
      Fix one starting date (TRAJECTORY_INDEX) and plot the entire forecast
      path against what actually happened.
      "Standing on this date, what did each model think the next h days
      would look like?"
      Good for showing long-horizon behaviour, e.g. a model that flattens
      out to a constant.

--------------------------------------------------------------------------
WHERE THE DATA COMES FROM
--------------------------------------------------------------------------
Reads the .npz files written by driver.py:
    results/preds/         LSTM, naive, and the TensorFlow KAN
    results/preds_pykan/   the pykan KAN
Each file holds predictions, ground truth, regime labels and dates for one
(model, config, horizon, seed, fold). Run driver.py first if these are absent.
"""

import glob
import json
import os

import numpy as np

import driver

import matplotlib
matplotlib.use("Agg")          # write files without needing a display
import matplotlib.pyplot as plt

# ==========================================================================
# CONFIG - edit this block
# ==========================================================================

HORIZON = 1                    # 1, 2, 100 or 200
SEED = 0                       # 0, 100 or 200
FOLDS = "last"                 # "last", "all", or a list such as [3, 4]

MODE = "step"                  # "step" or "trajectory"
STEP = 1                       # MODE="step": which step ahead, 1..HORIZON
TRAJECTORY_INDEX = -1          # MODE="trajectory": which test window (-1 = last)

KAN_SOURCE = "pykan"           # "pykan" or "custom"
SHOW_LSTM = True
SHOW_KAN = True
SHOW_NAIVE = False             # persistence baseline

TITLE = None                   # None = generated automatically
OUTPUT_DIR = "figures"
FILE_STEM = None               # None = generated automatically
WRITE_TEX = True               # also emit a pgfplots fragment for the paper
FIGSIZE = (8.6, 3.4)
DPI = 150

# Colours match the other figures in the paper.
COLOUR_ACTUAL = "#333333"
COLOUR_LSTM = "#2f6f9f"
COLOUR_KAN = "#c46a3f"
COLOUR_NAIVE = "#9a9a9a"

# ==========================================================================
# Everything below is machinery. You should not need to change it.
# ==========================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(HERE, "results", "preds")
PRED_DIR_PYKAN = os.path.join(HERE, "results", "preds_pykan")


def _fold_number(path):
    return int(os.path.basename(path).split("__f")[-1].split(".")[0])


def load_run(model, horizon, seed, folds, kan_source="pykan"):
    """
    Collect the fold files for one model at one horizon and seed, in date order.

    Returns (prediction, truth, dates) with prediction and truth of shape
    (n_windows, horizon), or None if no files are found.
    """
    if model == "kan" and kan_source == "pykan":
        pattern = os.path.join(PRED_DIR_PYKAN, "kan__-__h%d__s%d__f*.npz" % (horizon, seed))
    elif model == "kan":
        pattern = os.path.join(PRED_DIR, "kan__-__h%d__s%d__f*.npz" % (horizon, seed))
    elif model == "naive":
        # the baseline is deterministic, so it is only stored for seed 0
        pattern = os.path.join(PRED_DIR, "naive__-__h%d__s0__f*.npz" % horizon)
    elif model == "lstm":
        # Pin the config driver.py actually uses for this horizon. A bare "*"
        # here would also match the single-seed cross-check config that
        # driver.py runs at some horizons (see PRIMARY_LSTM_CONFIG vs
        # CONFIG_CHECK), silently mixing predictions from two architectures.
        config_name = driver.PRIMARY_LSTM_CONFIG[horizon]
        pattern = os.path.join(PRED_DIR, "lstm__%s__h%d__s%d__f*.npz" % (config_name, horizon, seed))
    else:
        raise ValueError("unknown model %r" % model)

    paths = sorted(glob.glob(pattern), key=_fold_number)
    if not paths:
        return None

    if folds == "last":
        paths = paths[-1:]
    elif folds != "all":
        wanted = set(folds)
        paths = [p for p in paths if _fold_number(p) in wanted]
        if not paths:
            return None

    predictions, truths, dates, config = [], [], [], None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            predictions.append(data["prediction"].astype(float))
            truths.append(data["truth"].astype(float))
            dates.append(data["dates"])
            config = json.loads(str(data["meta"]))["config"]

    return (
        np.concatenate(predictions),
        np.concatenate(truths),
        np.concatenate(dates),
        config,
    )


def build_series(run, mode, step, trajectory_index, horizon):
    """Reduce (n_windows, horizon) arrays to the single line we want to draw."""
    prediction, truth, dates, _ = run

    if mode == "step":
        if not 1 <= step <= horizon:
            raise ValueError("STEP must be between 1 and HORIZON (%d)" % horizon)
        index = step - 1
        return prediction[:, index], truth[:, index], dates

    if mode == "trajectory":
        row = trajectory_index if trajectory_index >= 0 else len(prediction) + trajectory_index
        if not 0 <= row < len(prediction):
            raise ValueError("TRAJECTORY_INDEX out of range (0..%d)" % (len(prediction) - 1))
        labels = np.array(["+%d" % (i + 1) for i in range(horizon)])
        return prediction[row], truth[row], labels

    raise ValueError("MODE must be 'step' or 'trajectory'")


def make_labels(dates, mode):
    """Sparse x tick positions and labels, so the axis stays readable."""
    count = len(dates)
    every = max(1, count // 6)
    positions = list(range(0, count, every))
    if mode == "step":
        labels = [str(dates[i])[:7] for i in positions]      # YYYY-MM
    else:
        labels = [str(dates[i]) for i in positions]          # +1, +21, ...
    return positions, labels


def main():
    series = {}

    if SHOW_LSTM:
        run = load_run("lstm", HORIZON, SEED, FOLDS)
        if run is None:
            print("no LSTM runs found for horizon %d seed %d" % (HORIZON, SEED))
        else:
            series["LSTM"] = build_series(run, MODE, STEP, TRAJECTORY_INDEX, HORIZON)

    if SHOW_KAN:
        run = load_run("kan", HORIZON, SEED, FOLDS, KAN_SOURCE)
        if run is None:
            print("no KAN runs found for horizon %d seed %d (source %s)"
                  % (HORIZON, SEED, KAN_SOURCE))
        else:
            series["KAN"] = build_series(run, MODE, STEP, TRAJECTORY_INDEX, HORIZON)

    if SHOW_NAIVE:
        run = load_run("naive", HORIZON, SEED, FOLDS)
        if run is not None:
            series["Naive"] = build_series(run, MODE, STEP, TRAJECTORY_INDEX, HORIZON)

    if not series:
        raise SystemExit(
            "Nothing to plot. Run driver.py first, or check HORIZON and SEED."
        )

    # ground truth should be identical across models, so take it from any of them
    any_key = next(iter(series))
    _, truth, dates = series[any_key]

    for name, (_, other_truth, _) in series.items():
        if len(other_truth) != len(truth) or not np.allclose(other_truth, truth):
            raise SystemExit(
                "%s was scored on %d windows, %s on %d, and the truth values "
                "do not match. Re-run driver.py so every model has finished "
                "the same folds for this HORIZON/SEED before plotting."
                % (name, len(other_truth), any_key, len(truth))
            )

    # ---------------------------------------------------------------- plot
    figure, axis = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(truth))

    axis.plot(x, truth, color=COLOUR_ACTUAL, linewidth=1.3, label="Actual",
              zorder=3)
    colours = {"LSTM": COLOUR_LSTM, "KAN": COLOUR_KAN, "Naive": COLOUR_NAIVE}
    for name, (prediction, _, _) in series.items():
        axis.plot(x, prediction, color=colours[name], linewidth=0.9, alpha=0.9,
                  label=name)

    positions, labels = make_labels(dates, MODE)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylabel("Close price (scaled)")
    if MODE == "trajectory":
        axis.set_xlabel("Days ahead")
    axis.grid(alpha=0.3, linewidth=0.4, linestyle="--")
    axis.legend(frameon=False, fontsize=8, ncol=len(series) + 1, loc="upper left")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

    if MODE == "step":
        default_title = "%d-day horizon, %d step(s) ahead" % (HORIZON, STEP)
    else:
        default_title = "%d-day forecast path from a single date" % HORIZON
    axis.set_title(TITLE or default_title, fontsize=10)

    figure.tight_layout()

    output_dir = os.path.join(HERE, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    mode_tag = ("step%d" % STEP) if MODE == "step" else ("traj%d" % TRAJECTORY_INDEX)
    kan_tag = ("_%s" % KAN_SOURCE) if SHOW_KAN else ""
    stem = FILE_STEM or ("figure_h%d_%s_s%d%s" % (HORIZON, mode_tag, SEED, kan_tag))
    png_path = os.path.join(output_dir, stem + ".png")
    figure.savefig(png_path, dpi=DPI)
    plt.close(figure)
    print("wrote %s" % png_path)

    # ------------------------------------------------------ pgfplots output
    if WRITE_TEX:
        tex_path = os.path.join(output_dir, stem + ".tex")
        with open(tex_path, "w") as handle:
            handle.write(build_tex(series, truth, dates, positions, labels, stem))
        print("wrote %s" % tex_path)

    # ------------------------------------------------------------- summary
    print()
    print("windows plotted: %d" % len(truth))
    if MODE == "step":
        print("date range: %s to %s" % (str(dates[0]), str(dates[-1])))
    for name, (prediction, own_truth, _) in series.items():
        error = float(np.sqrt(np.mean((prediction - own_truth) ** 2)))
        print("%-6s RMSE on this line: %.4f" % (name, error))


def build_tex(series, truth, dates, positions, labels, stem):
    """A pgfplots fragment in the same style as the paper's other figures."""
    tex_colours = {"LSTM": "blue!70!black", "KAN": "orange!80!black",
                   "Naive": "gray"}

    def coords(values):
        return " ".join("(%d,%.4f)" % (i, v) for i, v in enumerate(values))

    lines = [
        "% Generated by make_figure.py - do not edit by hand.",
        "\\begin{figure}[!ht]",
        "\\centering",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "  width=0.95\\textwidth, height=6.2cm,",
        "  xmin=0, xmax=%d, ylabel={Close price (scaled)}," % (len(truth) - 1),
        "  xtick={%s}," % ",".join(str(p) for p in positions),
        "  xticklabels={%s}," % ",".join(labels),
        "  x tick label style={font=\\scriptsize},",
        "  y tick label style={font=\\scriptsize},",
        "  legend style={font=\\scriptsize, draw=none}, legend pos=north west,",
        "  legend columns=%d, ymajorgrids, grid style={dashed,gray!30}]" % (len(series) + 1),
        "\\addplot+[mark=none, thick, black] coordinates {",
        coords(truth) + "};",
    ]
    for name, (prediction, _, _) in series.items():
        lines.append("\\addplot+[mark=none, semithick, %s] coordinates {" % tex_colours[name])
        lines.append(coords(prediction) + "};")
    lines.append("\\legend{Actual, %s}" % ", ".join(series.keys()))
    lines.append("\\end{axis}")
    lines.append("\\end{tikzpicture}")
    lines.append("\\caption{TODO write caption}")
    lines.append("\\label{fig:%s}" % stem)
    lines.append("\\end{figure}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
