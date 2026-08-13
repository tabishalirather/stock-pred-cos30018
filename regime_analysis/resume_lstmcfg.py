"""
resume_lstmcfg.py
Checkpoint-resumable runner for single Table 4 LSTM-grid jobs whose 25 epochs
of training do not fit inside one short shell invocation.

Training is split into slices of a few epochs. After each slice the full model
(weights and RMSprop optimizer state) is saved with Keras's native format, so
a resumed run continues the same optimisation trajectory. The one departure
from an uninterrupted run is the random shuffling of batches, which restarts
with the process; weight initialisation and the optimizer state are preserved.
This is recorded in the output metadata as resumed=True.

    python resume_lstmcfg.py --config L6-100-linear --seed 100 --fold 3
"""

import argparse
import json
import os
import time

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import driver  # noqa: E402
import regime_lib as rl  # noqa: E402

CKPT_DIR = os.path.join(driver.RESULTS_DIR, "ckpt_lstm")
TOTAL_EPOCHS = driver.EPOCHS  # 25, matching the protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs-per-slice", type=int, default=8)
    args = parser.parse_args()

    import tensorflow as tf

    horizon = driver.LSTM_GRID_HORIZON
    job = ("lstmcfg", args.config, horizon, args.seed, args.fold)
    out_path = driver.job_path(job)
    if os.path.exists(out_path):
        print("DONE (already on disk)")
        return

    spec = driver.LSTM_GRID[args.config]
    dataframe = rl.load_prices()
    x_all, y_all, dates = rl.build_sequences(dataframe, horizon)
    volatility, efficiency = rl.window_statistics(x_all)
    folds = rl.walk_forward_folds(len(x_all))
    train_slice, test_slice = folds[args.fold]

    x_train, y_train = x_all[train_slice], y_all[train_slice]
    x_test, y_test = x_all[test_slice], y_all[test_slice]
    thresholds = rl.regime_thresholds(volatility[train_slice], efficiency[train_slice])
    labels = rl.assign_regimes(volatility[test_slice], efficiency[test_slice], thresholds)

    os.makedirs(CKPT_DIR, exist_ok=True)
    stem = "%s_s%d_f%d" % (args.config, args.seed, args.fold)
    model_path = os.path.join(CKPT_DIR, stem + ".keras")
    state_path = os.path.join(CKPT_DIR, stem + ".json")

    epochs_done = 0
    elapsed_before = 0.0
    if os.path.exists(model_path) and os.path.exists(state_path):
        try:
            with open(state_path) as handle:
                state = json.load(handle)
            model = tf.keras.models.load_model(model_path)
            epochs_done = state["epochs"]
            elapsed_before = state["elapsed"]
            print("resuming at epoch %d" % epochs_done)
        except Exception as exc:
            print("checkpoint unreadable (%s); restarting" % type(exc).__name__)
            epochs_done = 0
            model = None
    else:
        model = None

    if model is None:
        model = driver.build_lstm(spec, x_train.shape[1], x_train.shape[2],
                                  horizon, args.seed + args.fold)

    started = time.time()
    to_run = min(args.epochs_per_slice, TOTAL_EPOCHS - epochs_done)
    if to_run > 0:
        model.fit(x_train, y_train, epochs=to_run,
                  batch_size=driver.BATCH_SIZE, verbose=0)
        epochs_done += to_run
    elapsed_total = elapsed_before + (time.time() - started)

    if epochs_done < TOTAL_EPOCHS:
        model.save(model_path)
        with open(state_path, "w") as handle:
            json.dump({"epochs": epochs_done, "elapsed": elapsed_total}, handle)
        print("CHECKPOINT at epoch %d/%d" % (epochs_done, TOTAL_EPOCHS))
        return

    prediction = model.predict(x_test, verbose=0)
    train_rmse = rl.rmse(model.predict(x_train, verbose=0), y_train)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        prediction=np.asarray(prediction, dtype=np.float32),
        truth=np.asarray(y_test, dtype=np.float32),
        labels=np.asarray(labels, dtype="U8"),
        dates=np.asarray([np.datetime_as_string(d, unit="D") for d in dates[test_slice]]),
        meta=json.dumps({
            "model": "lstmcfg", "config": args.config, "horizon": horizon,
            "seed": args.seed, "fold": args.fold,
            "n_train": int(len(x_train)), "n_test": int(len(x_test)),
            "parameters": int(model.count_params()),
            "thresholds": thresholds,
            "train_seconds": round(elapsed_total, 3),
            "train_rmse": train_rmse,
            "resumed": True,
        }),
    )
    for path in (model_path, state_path):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
    print("DONE wrote %s" % os.path.basename(out_path))


if __name__ == "__main__":
    main()
