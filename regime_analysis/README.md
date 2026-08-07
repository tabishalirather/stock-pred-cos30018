# regime_analysis

Regime-conditional re-evaluation of the KAN vs LSTM comparison, built to replace
Table 7 of the Operations Research Forum submission with measured results.

Read `FINDINGS.md` first — it explains what the analysis found and why some of
the published numbers do not reproduce.

## Reproducing

```bash
python driver.py --status          # show how many of the 150 runs are done
python driver.py --budget 600      # run jobs until the budget is used up
python aggregate.py                # pool folds, write table + figure + summary
python replicate_published.py      # rerun the ORIGINAL protocol for comparison
```

`driver.py` is resumable: it writes one `.npz` per (model, config, horizon, seed,
fold) into `results/preds/` and skips anything already there, so it can be run in
short bursts or restarted after an interruption. `--budget` is a wall-clock
allowance in seconds and `--workers N --worker i` splits the grid across
processes.

The grid is 150 runs: naive baseline (4 horizons x 5 folds), KAN
(4 x 3 seeds x 5 folds), LSTM at its per-horizon configuration
(4 x 3 x 5) and a single-seed configuration cross-check at 1 and 2 days.

## Protocol summary

| | Published pipeline | Here |
|---|---|---|
| Split | `train_test_split`, shuffled, no seed | blocked walk-forward, 5 folds |
| Test set | LSTM: all of it. KAN: last `h` rows | identical windows for both |
| Regimes | undefined | volatility + efficiency ratio, train-only thresholds |
| Baseline | none | persistence (repeat last close) |
| 200-day | LSTM not run | both models run |
| Seeds | none fixed | 0, 100, 200 |

Regime rule, applied to the 20-day input window only (no look-ahead):

* `Volatile` if window volatility exceeds the training-set 75th percentile
* `Trending` else if Kaufman efficiency ratio exceeds the training-set 75th percentile
* `Normal` otherwise

## KAN engines

The KAN grid exists twice, and the two engines corroborate each other (see
`FINDINGS.md` section 4):

* `results/preds/` — `kan_layer.py`, a TensorFlow reimplementation of pykan's
  construction (run with the default `--kan-engine custom`)
* `results/preds_pykan/` — **the real pykan package** (pykan 0.2.8,
  torch 2.2.2), run with `--kan-engine pykan`; summary in
  `results/summary_pykan.json`

The paper should quote the pykan numbers, with the fold-0 L-BFGS divergence at
long horizons disclosed (documented in `FINDINGS.md`). The LSTM half uses Keras
exactly as the original did in both cases.

## Requirements

Python 3.10, `numpy`, `pandas`, `scikit-learn`, `matplotlib`,
`tensorflow-cpu==2.16.2`, plus `pykan==0.2.8` and `torch==2.2.2` for the pykan
engine.

Data is read from `../v0.2/MLP/data/CBA.AX_2020-01-01_2023-08-01.csv`; override
with the `REGIME_DATA_CSV` environment variable.
