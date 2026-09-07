"""
provenance.py
Answer the question "are these results actually mine, or the ones that came
with the repository?"

The repository ships a populated results/ directory containing the runs behind
the published tables. driver.py skips any job whose output file already exists,
so a fresh clone run with the default settings finds everything done and writes
nothing. That is convenient for resuming an interrupted grid and misleading for
anyone trying to reproduce the work.

This script reports, for a results directory:

  * how many prediction files it holds
  * whether each file is tracked by git at its committed content, which means
    it came with the clone, or is new or modified, which means it was written
    locally
  * the recorded wall-clock training time, which is machine specific and will
    not match another person's run even when the predictions agree

    python provenance.py                      # inspect results/
    python provenance.py --dir results_mine   # inspect an independent re-run
    python provenance.py --dir results_mine --compare results

Comparing two directories reports, per job, whether the predictions agree.
Runs are seeded, so an honest independent re-run should agree closely. Small
differences in the last decimal places are normal across machines and library
builds. Large differences are the interesting case and are listed in full.
"""

import argparse
import glob
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUBDIRS = ["preds", "preds_pykan", "preds_kangrid", "preds_lstmgrid"]


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def git_status(paths):
    """
    Split paths into those git considers unchanged from the commit and those
    it reports as new or modified. Returns (unchanged, local, available).
    """
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=HERE, capture_output=True, text=True, timeout=30,
        )
        if top.returncode != 0:
            return set(), set(), False
        root = top.stdout.strip()
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--"] + paths,
            cwd=root, capture_output=True, text=True, timeout=120,
        )
        if out.returncode != 0:
            return set(), set(), False
    except (OSError, subprocess.SubprocessError):
        return set(), set(), False

    local = set()
    for line in out.stdout.splitlines():
        if len(line) > 3:
            local.add(os.path.normpath(os.path.join(root, line[3:].strip('"'))))
    unchanged = {os.path.normpath(p) for p in paths} - local
    return unchanged, local, True


def read_meta(path):
    try:
        with np.load(path, allow_pickle=False) as data:
            return json.loads(str(data["meta"]))
    except Exception:
        return {}


def inspect(directory):
    directory = resolve(directory)
    if not os.path.isdir(directory):
        print("no such directory: %s" % directory)
        return None

    files = []
    for sub in SUBDIRS:
        files.extend(sorted(glob.glob(os.path.join(directory, sub, "*.npz"))))
    if not files:
        print("%s holds no prediction files" % directory)
        return []

    unchanged, local, have_git = git_status(files)

    print("directory        : %s" % directory)
    print("prediction files : %d" % len(files))
    if have_git:
        print("came with clone  : %d" % len(unchanged))
        print("written locally  : %d" % len(local))
        if not local:
            print()
            print("  Every file here is byte for byte what the repository shipped.")
            print("  Nothing in this directory was computed on this machine.")
            print("  Re-run with REGIME_RESULTS_DIR pointing at an empty path.")
        elif not unchanged:
            print()
            print("  Every file here was written locally.")
    else:
        print("git              : unavailable, cannot separate shipped from local")

    seconds = [m.get("train_seconds") for m in map(read_meta, files)]
    seconds = [s for s in seconds if isinstance(s, (int, float))]
    if seconds:
        print("total train time : %.1f s across %d runs (machine specific)"
              % (sum(seconds), len(seconds)))
    return files


def key_of(path):
    return os.path.join(os.path.basename(os.path.dirname(path)),
                        os.path.basename(path))


def compare(dir_a, dir_b, tolerance):
    a_files, b_files = {}, {}
    for store, directory in ((a_files, dir_a), (b_files, dir_b)):
        for sub in SUBDIRS:
            for path in glob.glob(os.path.join(resolve(directory), sub, "*.npz")):
                store[key_of(path)] = path

    shared = sorted(set(a_files) & set(b_files))
    print()
    print("comparing %s against %s" % (dir_a, dir_b))
    print("jobs in both     : %d" % len(shared))
    print("only in %-8s : %d" % (dir_a[:8], len(set(a_files) - set(b_files))))
    print("only in %-8s : %d" % (dir_b[:8], len(set(b_files) - set(a_files))))
    if not shared:
        return

    identical, close, differing = 0, 0, []
    for k in shared:
        try:
            with np.load(a_files[k], allow_pickle=False) as da, \
                 np.load(b_files[k], allow_pickle=False) as db:
                pa = da["prediction"].astype(float)
                pb = db["prediction"].astype(float)
        except Exception:
            differing.append((k, float("nan")))
            continue
        if pa.shape != pb.shape:
            differing.append((k, float("nan")))
            continue
        gap = float(np.max(np.abs(pa - pb))) if pa.size else 0.0
        if gap == 0.0:
            identical += 1
        elif gap <= tolerance:
            close += 1
        else:
            differing.append((k, gap))

    print("identical        : %d" % identical)
    print("agree within %-4g: %d" % (tolerance, close))
    print("differ           : %d" % len(differing))
    for k, gap in sorted(differing, key=lambda x: -(x[1] if x[1] == x[1] else 0))[:20]:
        print("   %-52s max abs diff %s" % (k, "shape/read error" if gap != gap else "%.6g" % gap))
    if differing:
        print()
        print("  Differences above the tolerance are worth investigating before")
        print("  the numbers are relied on. Report them rather than averaging them away.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results",
                        help="results directory to inspect (default: results)")
    parser.add_argument("--compare", default=None,
                        help="second results directory to compare against")
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="max absolute prediction difference treated as agreement")
    args = parser.parse_args()

    inspect(args.dir)
    if args.compare:
        compare(args.dir, args.compare, args.tolerance)


if __name__ == "__main__":
    main()
