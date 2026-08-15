"""
resume_kancfg.py
Checkpoint-resumable runner for single Table 6 KAN-grid jobs whose training does
not fit inside one short shell invocation.

It replicates pykan's fit(opt="LBFGS", steps=10) loop exactly - same custom
LBFGS construction, same MSE objective, same grid updates at steps 0 and 5
(grid_update_freq = stop_grid_update_step / grid_update_num = 50/10 = 5) - but
saves model + optimizer state between invocations so the 10 optimisation steps
can be completed across several calls. The LBFGS history is preserved through
optimizer.state_dict(), so the resumed trajectory matches an uninterrupted run.

    python resume_kancfg.py --config C1 --seed 0 --fold 4 --max-seconds 30

Run repeatedly until it reports DONE; the final invocation writes the same .npz
that driver.py would have produced.
"""

import argparse
import os
import sys

# Pin hash randomisation before anything else: see the note in driver.py.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import json
import time

import numpy as np
import torch

import driver
import regime_lib as rl

CKPT_DIR = os.path.join(driver.RESULTS_DIR, "ckpt")


def build_job_state(config_name, seed, fold, horizon):
    from kan import KAN

    spec = driver.KAN_GRID[config_name]
    dataframe = rl.load_prices()
    x_all, y_all, dates = rl.build_sequences(dataframe, horizon)
    volatility, efficiency = rl.window_statistics(x_all)
    folds = rl.walk_forward_folds(len(x_all))
    train_slice, test_slice = folds[fold]

    x_train, y_train = x_all[train_slice], y_all[train_slice]
    x_test, y_test = x_all[test_slice], y_all[test_slice]
    thresholds = rl.regime_thresholds(volatility[train_slice], efficiency[train_slice])
    labels = rl.assign_regimes(volatility[test_slice], efficiency[test_slice], thresholds)

    # Seed numpy as well as torch: pykan's fit() uses np.random.choice to
    # permute the training rows each step. See the note in driver.py.
    np.random.seed(seed + fold)
    torch.manual_seed(seed + fold)
    flat_train = torch.tensor(x_train.reshape(len(x_train), -1), dtype=torch.float32)
    flat_test = torch.tensor(x_test.reshape(len(x_test), -1), dtype=torch.float32)
    hidden = max(3, len(flat_train) // spec["divisor"])
    net = KAN(width=[flat_train.shape[1], hidden, horizon], grid=spec["grid"],
              k=spec["k"], seed=seed + fold, device="cpu", auto_save=False)
    data = {
        "train_input": flat_train,
        "train_label": torch.tensor(y_train, dtype=torch.float32),
        "test_input": flat_test,
        "test_label": torch.tensor(y_test, dtype=torch.float32),
    }
    meta = {
        "x_train_n": int(len(x_train)),
        "labels": labels,
        "dates": dates[test_slice],
        "thresholds": thresholds,
        "y_train": y_train,
        "y_test": y_test,
    }
    return net, data, meta


def make_optimizer(net):
    from kan.LBFGS import LBFGS

    return LBFGS(net.get_params(), lr=1.0, history_size=10,
                 line_search_fn="strong_wolfe", tolerance_grad=1e-32,
                 tolerance_change=1e-32, tolerance_ys=1e-32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    args = parser.parse_args()

    horizon = driver.KAN_GRID_HORIZON
    job = ("kancfg", args.config, horizon, args.seed, args.fold)
    out_path = driver.job_path(job)
    if os.path.exists(out_path):
        print("DONE (already on disk)")
        return

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(
        CKPT_DIR, "%s_s%d_f%d.pt" % (args.config, args.seed, args.fold)
    )

    net, data, meta = build_job_state(args.config, args.seed, args.fold, horizon)
    optimizer = make_optimizer(net)

    step_done = 0
    elapsed_before = 0.0
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, weights_only=False)
            net.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            step_done = ckpt["step"]
            elapsed_before = ckpt["elapsed"]
        except Exception as exc:  # corrupt checkpoint (interrupted save)
            print("checkpoint unreadable (%s); restarting job" % type(exc).__name__)
            step_done = 0

    # Replicate fit()'s training-mode switches exactly: disable_symbolic_in_fit
    # turns off the per-edge symbolic branch (a Python loop that dominates the
    # forward pass if left on) and activation caching for lamb=0.
    net.disable_symbolic_in_fit(0.0)
    net.save_act = False
    loss_fn = lambda x, y: torch.mean((x - y) ** 2)

    def closure():
        optimizer.zero_grad()
        pred = net.forward(data["train_input"])
        loss = loss_fn(pred, data["train_label"])
        loss.backward()
        return loss

    started = time.time()
    while step_done < args.steps:
        if time.time() - started > args.max_seconds:
            break
        # pykan fit(): update_grid at steps where step % 5 == 0 (freq = 50/10)
        if step_done % 5 == 0:
            net.update_grid(data["train_input"])
        optimizer.step(closure)
        step_done += 1
        print("step %d/%d done (%.1fs)" % (step_done, args.steps,
                                           time.time() - started), flush=True)

    elapsed_total = elapsed_before + (time.time() - started)

    if step_done < args.steps:
        tmp_path = ckpt_path + ".tmp"
        torch.save({"model": net.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step_done, "elapsed": elapsed_total}, tmp_path)
        os.replace(tmp_path, ckpt_path)
        print("CHECKPOINT at step %d/%d" % (step_done, args.steps))
        return

    with torch.no_grad():
        prediction = net(data["test_input"]).cpu().numpy()
        train_prediction = net(data["train_input"]).cpu().numpy()
    train_rmse = rl.rmse(train_prediction, meta["y_train"])
    parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        prediction=np.asarray(prediction, dtype=np.float32),
        truth=np.asarray(meta["y_test"], dtype=np.float32),
        labels=np.asarray(meta["labels"], dtype="U8"),
        dates=np.asarray([np.datetime_as_string(d, unit="D") for d in meta["dates"]]),
        meta=json.dumps({
            "model": "kancfg", "config": args.config, "horizon": horizon,
            "seed": args.seed, "fold": args.fold,
            "n_train": meta["x_train_n"], "n_test": int(len(meta["y_test"])),
            "parameters": int(parameters), "thresholds": meta["thresholds"],
            "train_seconds": round(elapsed_total, 3),
            "train_rmse": train_rmse,
            "resumed": True,
        }),
    )
    try:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
    except OSError:
        pass  # stale checkpoints are ignored on future runs
    print("DONE wrote %s" % os.path.basename(out_path))


if __name__ == "__main__":
    main()
