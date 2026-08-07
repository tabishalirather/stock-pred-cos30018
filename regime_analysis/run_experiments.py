"""
run_experiments.py
Regime-conditional evaluation of LSTM and KAN on CBA.AX daily data.

For each horizon the script runs blocked walk-forward validation. Within every
fold it trains on all earlier sequences, predicts the contiguous test block,
labels each test window Normal / Volatile / Trending using thresholds taken from
that fold's training window only, then pools the predictions across folds and
reports RMSE overall and per regime. Repeating with several seeds gives the
mean and standard deviation quoted in the paper.

Usage
-----
    python run_experiments.py --model lstm --config 100u-linear --horizons 1 2
    python run_experiments.py --model kan  --horizons 1 2 100 200
    python run_experiments.py --model naive --horizons 1 2 100 200
"""

import argparse
import json
import os
import time

import numpy as np

import regime_lib as rl

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "regime_results.json")

# LSTM configurations exactly as specified in the published paper
LSTM_CONFIGS = {
    "100u-linear": {"units": 100, "activation": "linear", "layers": 4},
    "10u-tanh": {"units": 10, "activation": "tanh", "layers": 4},
}
EPOCHS = 25
BATCH_SIZE = 30
DROPOUT = 0.2


def build_lstm(config, lookback, n_features, horizon, seed):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    layers = [tf.keras.layers.Input(shape=(lookback, n_features))]
    for index in range(config["layers"]):
        is_last = index == config["layers"] - 1
        layers.append(tf.keras.layers.LSTM(config["units"], return_sequences=not is_last))
        layers.append(tf.keras.layers.Dropout(DROPOUT))
    layers.append(tf.keras.layers.Dense(horizon, activation=config["activation"]))
    model = tf.keras.Sequential(layers)
    model.compile(loss="mean_squared_error", optimizer="RMSprop", metrics=["mean_squared_error"])
    return model


def fit_predict_lstm(config, x_train, y_train, x_test, horizon, seed):
    model = build_lstm(config, x_train.shape[1], x_train.shape[2], horizon, seed)
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    return model.predict(x_test, verbose=0), model.count_params()


def fit_predict_kan(x_train, y_train, x_test, horizon, seed):
    import tensorflow as tf

    from kan_layer import build_kan, fit_lbfgs

    tf.keras.utils.set_random_seed(seed)
    flat_train = x_train.reshape(len(x_train), -1)
    flat_test = x_test.reshape(len(x_test), -1)

    # same heuristic as the published KAN script: neurons = len(train) // 50,
    # hidden width = 3 * neurons
    hidden = max(3, 3 * (len(flat_train) // 50))
    model = build_kan(flat_train.shape[1], hidden, horizon, grid_size=3, spline_order=3, seed=seed)
    fit_lbfgs(model, flat_train, y_train, max_iterations=10)
    return model.predict(flat_test, verbose=0), model.count_params()


def run_one(model_name, config_name, horizon, seed, dataframe):
    x_all, y_all, dates = rl.build_sequences(dataframe, horizon)
    volatility, efficiency = rl.window_statistics(x_all)
    folds = rl.walk_forward_folds(len(x_all))

    pooled_pred, pooled_true, pooled_labels, pooled_dates = [], [], [], []
    fold_notes = []
    parameters = None

    for fold_index, (train_slice, test_slice) in enumerate(folds):
        x_train, y_train = x_all[train_slice], y_all[train_slice]
        x_test, y_test = x_all[test_slice], y_all[test_slice]

        thresholds = rl.regime_thresholds(volatility[train_slice], efficiency[train_slice])
        labels = rl.assign_regimes(volatility[test_slice], efficiency[test_slice], thresholds)

        if model_name == "lstm":
            prediction, parameters = fit_predict_lstm(
                LSTM_CONFIGS[config_name], x_train, y_train, x_test, horizon, seed + fold_index
            )
        elif model_name == "kan":
            prediction, parameters = fit_predict_kan(
                x_train, y_train, x_test, horizon, seed + fold_index
            )
        elif model_name == "naive":
            prediction, parameters = rl.naive_last_value(x_test, horizon), 0
        else:
            raise ValueError("unknown model %s" % model_name)

        pooled_pred.append(np.asarray(prediction, dtype=np.float64))
        pooled_true.append(np.asarray(y_test, dtype=np.float64))
        pooled_labels.append(labels)
        pooled_dates.append(dates[test_slice])
        fold_notes.append(
            {
                "fold": fold_index,
                "n_train": int(len(x_train)),
                "n_test": int(len(x_test)),
                "thresholds": thresholds,
                "test_start": str(np.datetime_as_string(dates[test_slice][0], unit="D")),
                "test_end": str(np.datetime_as_string(dates[test_slice][-1], unit="D")),
            }
        )

    prediction = np.concatenate(pooled_pred)
    truth = np.concatenate(pooled_true)
    labels = np.concatenate(pooled_labels)

    # a label applies to a whole multi-step target row, so repeat it per step
    step_labels = np.repeat(labels[:, None], horizon, axis=1).ravel()
    metrics = rl.rmse_by_regime(prediction.ravel(), truth.ravel(), step_labels)

    return {
        "model": model_name,
        "config": config_name,
        "horizon": horizon,
        "seed": seed,
        "parameters": int(parameters or 0),
        "n_test_windows": int(len(labels)),
        "metrics": metrics,
        "folds": fold_notes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["lstm", "kan", "naive"])
    parser.add_argument("--config", default="-")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 100, 200])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 100, 200])
    parser.add_argument("--out", default=RESULTS_PATH)
    args = parser.parse_args()

    if args.model == "lstm" and args.config not in LSTM_CONFIGS:
        raise SystemExit("--config must be one of %s" % list(LSTM_CONFIGS))

    dataframe = rl.load_prices()
    seeds = [0] if args.model == "naive" else args.seeds

    for horizon in args.horizons:
        for seed in seeds:
            started = time.time()
            record = run_one(args.model, args.config, horizon, seed, dataframe)
            record["seconds"] = round(time.time() - started, 2)
            rl.append_result(record, args.out)
            summary = record["metrics"]
            print(
                "%-5s %-12s h=%-3d seed=%-4d overall=%.4f normal=%.4f volatile=%.4f "
                "trending=%.4f (%.1fs)"
                % (
                    args.model,
                    args.config,
                    horizon,
                    seed,
                    summary["Overall"]["rmse"],
                    summary["Normal"]["rmse"],
                    summary["Volatile"]["rmse"],
                    summary["Trending"]["rmse"],
                    record["seconds"],
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
