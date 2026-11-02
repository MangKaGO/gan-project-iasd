"""Evaluate every saved checkpoint of every training run (vanilla sampling).

    python experiments/eval_training.py --runs_dir runs

Output: results/<features>/training_metrics.csv, one line per (run, epoch) with
FID, precision, recall and class statistics. Already evaluated checkpoints are
skipped, so the script can be relaunched safely after new runs finish.
"""
import argparse
import glob
import os
import re

from common import append_row, list_runs, read_rows, run_info

import torch  # noqa: E402

from evaluation.metrics import Evaluator  # noqa: E402
from sampling import sample  # noqa: E402
from utils import get_device, load_generator, set_seed  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--features", default="platform", choices=["platform", "mnist"],
                        help="platform: Inception FID + VGG-16 P/R (as the course platform); "
                             "mnist: cheap MNIST CNN features.")
    parser.add_argument("--out", default=None,
                        help="Default: results/<features>/training_metrics.csv")
    parser.add_argument("--n", type=int, default=10000, help="Number of generated images.")
    parser.add_argument("--eval_seed", type=int, default=0, help="Seed of the latent samples.")
    args = parser.parse_args()
    args.out = args.out or os.path.join("results", args.features, "training_metrics.csv")

    device = get_device()
    evaluator = Evaluator(device, data_path=os.getenv("DATA"), features=args.features)
    done = {(r["run"], int(r["epoch"])) for r in read_rows(args.out)}

    for run_dir in list_runs(args.runs_dir):
        info = run_info(run_dir)
        ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "G_epoch*.pth")))
        for path in ckpts:
            epoch = int(re.search(r"epoch(\d+)", path).group(1))
            if (info["run"], epoch) in done:
                continue
            G = load_generator(os.path.dirname(path), device, os.path.basename(path))
            set_seed(args.eval_seed)
            with torch.no_grad():
                images, _ = sample(G, args.n, device, method="vanilla")
            metrics = evaluator(images)
            row = {**info, "epoch": epoch, **{k: round(v, 5) for k, v in metrics.items()}}
            append_row(args.out, row)
            print(row)


if __name__ == "__main__":
    main()
