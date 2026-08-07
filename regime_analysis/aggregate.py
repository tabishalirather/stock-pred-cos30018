"""
aggregate.py
Turn the per-fold prediction files written by driver.py into the summary
statistics, the LaTeX table and the comparison figures used in the paper.

For every (model, config, horizon, seed) the fold predictions are pooled, then
RMSE is computed overall and per regime. The mean and sample standard deviation
across seeds is what gets reported. Every number the table prints is traceable
to the .npz files in results/preds/.

    python aggregate.py
"""

import glob
import json
import os
from collections import defaultdict

import numpy as np

import regime_lib as rl

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
PRED_DIR = os.path.join(RESULTS_DIR, "preds")
REGIMES = rl.REGIMES
HORIZON_LABELS = {1: "1-Day", 2: "2-Day", 100: "100-Day", 200: "200-Day"}


# --------------------------------------------------------------------------
def load_runs():
    """group fold files into runs keyed by (model, config, horizon, seed)"""
    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(PRED_DIR, "*.npz"))):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            runs[(meta["model"], meta["config"], meta["horizon"], meta["seed"])].append(
                {
                    "prediction": data["prediction"],
                    "truth": data["truth"],
                    "labels": data["labels"],
                    "dates": data["dates"],
                    "meta": meta,
                }
            )
    return runs


def summarise_run(folds):
    """pool a run's folds and return RMSE overall and per regime"""
    folds = sorted(folds, key=lambda item: item["meta"]["fold"])
    horizon = folds[0]["meta"]["horizon"]
    prediction = np.concatenate([f["prediction"] for f in folds]).astype(np.float64)
    truth = np.concatenate([f["truth"] for f in folds]).astype(np.float64)
    labels = np.concatenate([f["labels"] for f in folds])
    step_labels = np.repeat(labels[:, None], horizon, axis=1).ravel()
    metrics = rl.rmse_by_regime(prediction.ravel(), truth.ravel(), step_labels)
    metrics["_windows"] = {
        regime: int((labels == regime).sum()) for regime in REGIMES
    }
    metrics["_n_folds"] = len(folds)
    metrics["_dates"] = (str(folds[0]["dates"][0]), str(folds[-1]["dates"][-1]))
    metrics["_parameters"] = folds[0]["meta"]["parameters"]
    return metrics


def build_summary():
    runs = load_runs()
    per_run = {key: summarise_run(folds) for key, folds in runs.items()}

    summary = defaultdict(dict)
    seeds_seen = defaultdict(list)
    for (model, config, horizon, seed), metrics in per_run.items():
        seeds_seen[(model, config, horizon)].append((seed, metrics))

    for (model, config, horizon), entries in seeds_seen.items():
        entries.sort()
        block = {"seeds": [seed for seed, _ in entries], "n_seeds": len(entries)}
        for column in ["Overall"] + REGIMES:
            values = [metrics[column]["rmse"] for _, metrics in entries]
            counts = [metrics[column]["n"] for _, metrics in entries]
            block[column] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "n_points": int(counts[0]),
            }
        block["windows"] = entries[0][1]["_windows"]
        block["dates"] = entries[0][1]["_dates"]
        block["parameters"] = entries[0][1]["_parameters"]
        summary[model + "|" + config][horizon] = block
    return per_run, summary


# --------------------------------------------------------------------------
def latex_table(summary):
    """Rebuild Table 7 from measured results."""
    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{|l|l|c|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Horizon} & \textbf{Regime} & \textbf{Windows} & \textbf{LSTM RMSE} & "
        r"\textbf{KAN RMSE} & \textbf{Naive RMSE} & \textbf{LSTM vs KAN} \\"
    )
    lines.append(r"\hline")

    for horizon in sorted(HORIZON_LABELS):
        lstm_key = next(
            (key for key in summary if key.startswith("lstm|") and horizon in summary[key]
             and summary[key][horizon]["n_seeds"] > 1),
            None,
        )
        if lstm_key is None or horizon not in summary.get("kan|-", {}):
            continue
        lstm = summary[lstm_key][horizon]
        kan = summary["kan|-"][horizon]
        naive = summary["naive|-"][horizon]
        config = lstm_key.split("|")[1]

        for index, regime in enumerate(REGIMES):
            label = (
                r"\multirow{3}{*}{%s}" % HORIZON_LABELS[horizon] if index == 0 else ""
            )
            ratio = kan[regime]["mean"] / lstm[regime]["mean"]
            lines.append(
                "  %s & %s & %d & $%.3f \\pm %.3f$ & $%.3f \\pm %.3f$ & %.3f & "
                "%.1f$\\times$ \\\\"
                % (
                    label,
                    regime,
                    lstm["windows"][regime],
                    lstm[regime]["mean"],
                    lstm[regime]["std"],
                    kan[regime]["mean"],
                    kan[regime]["std"],
                    naive[regime]["mean"],
                    ratio,
                )
            )
        overall_ratio = kan["Overall"]["mean"] / lstm["Overall"]["mean"]
        lines.append(
            "  & \\textit{All} & %d & $%.3f \\pm %.3f$ & $%.3f \\pm %.3f$ & %.3f & "
            "%.1f$\\times$ \\\\"
            % (
                sum(lstm["windows"].values()),
                lstm["Overall"]["mean"],
                lstm["Overall"]["std"],
                kan["Overall"]["mean"],
                kan["Overall"]["std"],
                naive["Overall"]["mean"],
                overall_ratio,
            )
        )
        lines.append(r"\hline")
        lines.append("%% horizon %d used LSTM configuration %s" % (horizon, config))

    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Regime-conditional performance of LSTM and KAN across forecast "
        r"horizons (RMSE in Min-Max scaled space, lower is better). Regimes are "
        r"assigned from the 20-day input window using the rule defined in "
        r"Section~\ref{sec:regimes}; \emph{Windows} is the number of test windows "
        r"falling in each regime. Figures are the mean and sample standard deviation "
        r"over three random seeds, each pooled across five blocked walk-forward "
        r"folds. \emph{Naive} is the persistence baseline that repeats the last "
        r"observed close price. Both architectures are evaluated on identical test "
        r"windows at every horizon, including 200 days.}"
    )
    lines.append(r"\label{table_regime_comparison}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def config_check_note(summary):
    """Report the single-seed cross-check of the inherited LSTM configuration."""
    notes = []
    for key in sorted(summary):
        if not key.startswith("lstm|"):
            continue
        config = key.split("|")[1]
        for horizon, block in sorted(summary[key].items()):
            notes.append(
                "h=%-3d %-12s seeds=%d overall=%.4f"
                % (horizon, config, block["n_seeds"], block["Overall"]["mean"])
            )
    return "\n".join(notes)


# --------------------------------------------------------------------------
def make_figures(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = [h for h in sorted(HORIZON_LABELS) if h in summary.get("kan|-", {})]
    figure, axes = plt.subplots(1, len(horizons), figsize=(4.0 * len(horizons), 3.6),
                                sharey=False)
    if len(horizons) == 1:
        axes = [axes]

    width = 0.26
    positions = np.arange(len(REGIMES))
    for axis, horizon in zip(axes, horizons):
        lstm_key = next(
            key for key in summary if key.startswith("lstm|") and horizon in summary[key]
            and summary[key][horizon]["n_seeds"] > 1
        )
        lstm = summary[lstm_key][horizon]
        kan = summary["kan|-"][horizon]
        naive = summary["naive|-"][horizon]

        axis.bar(positions - width, [lstm[r]["mean"] for r in REGIMES], width,
                 yerr=[lstm[r]["std"] for r in REGIMES], capsize=3, label="LSTM",
                 color="#2f6f9f")
        axis.bar(positions, [kan[r]["mean"] for r in REGIMES], width,
                 yerr=[kan[r]["std"] for r in REGIMES], capsize=3, label="KAN",
                 color="#c46a3f")
        axis.bar(positions + width, [naive[r]["mean"] for r in REGIMES], width,
                 label="Naive", color="#9a9a9a")

        axis.set_xticks(positions)
        axis.set_xticklabels(REGIMES)
        axis.set_title("%s horizon" % HORIZON_LABELS[horizon])
        axis.set_ylabel("RMSE (scaled)")
        axis.grid(axis="y", alpha=0.3, linewidth=0.5)
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    output = os.path.join(RESULTS_DIR, "regime_comparison_grid.png")
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


# --------------------------------------------------------------------------
def main():
    per_run, summary = build_summary()

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as handle:
        json.dump(
            {"|".join(map(str, key)): value for key, value in
             {tuple(k.split("|")) + (h,): v for k, d in summary.items() for h, v in d.items()}.items()},
            handle,
            indent=2,
            default=float,
        )

    table = latex_table(summary)
    with open(os.path.join(RESULTS_DIR, "table_regime_comparison.tex"), "w") as handle:
        handle.write(table + "\n")

    figure_path = make_figures(summary)

    print("runs summarised:", len(per_run))
    print()
    print(config_check_note(summary))
    print()
    print(table)
    print()
    print("figure:", figure_path)


if __name__ == "__main__":
    main()
