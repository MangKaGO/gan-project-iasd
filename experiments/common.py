"""Helpers shared by the experiment scripts (run them from the repository root)."""
import csv
import glob
import os
import sys

# Make the repository root importable (model.py, utils.py, sampling.py, ...).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import load_config  # noqa: E402


def run_label(config):
    """Name of the training configuration of a run, from its config.json."""
    ttur = config.get("lr_d") != config.get("lr_g")
    sn = config.get("spectral_norm", False)
    if sn and ttur:
        return "sn_ttur"
    if sn:
        return "sn"
    return "ttur" if ttur else "baseline"


def is_finished(run_dir):
    """A run is finished when the checkpoint of its last epoch exists."""
    epochs = load_config(run_dir).get("epochs")
    return epochs is not None and os.path.exists(
        os.path.join(run_dir, "checkpoints", f"G_epoch{epochs:03d}.pth"))


def list_runs(runs_dir):
    """FINISHED run folders, sorted. Runs still training are skipped, so that a
    half-trained model is never evaluated as if it were final."""
    runs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(runs_dir, "*", "config.json")))
    for r in runs:
        if not is_finished(r):
            print(f"Skipping {r}: training not finished.")
    return [r for r in runs if is_finished(r)]


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def append_row(path, row):
    """Append one result line to a CSV file (header written on first use).
    Results are written immediately, so an interrupted job loses nothing."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def run_info(run_dir):
    config = load_config(run_dir)
    return {"run": os.path.basename(run_dir), "config": run_label(config),
            "seed": config.get("seed", 0)}
