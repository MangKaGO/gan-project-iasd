"""Compare sampling methods on the FINAL checkpoint of each run (inference lever).

    python experiments/eval_sampling.py --runs_dir runs

For each run:
  - vanilla                       (reference, budget K = 1)
  - truncation, psi in --psis     (no rejection: cost K = 1)
  - DRS and OBRS, K in --Ks       (same budget K for both: acceptance 1/K)
  - the same rejection samplers with D_calibrated.pth, if
    experiments/calibrate_D.py was run for this run (ablation "D quality").

Output: results/<features>/sampling_metrics.csv with FID, precision, recall, class
statistics, acceptance rate, number of generated candidates and total time
(calibration included). Already computed lines are skipped.
"""
import argparse
import os

from common import append_row, list_runs, read_rows, run_info

import torch  # noqa: E402

from evaluation.metrics import Evaluator  # noqa: E402
from sampling import sample  # noqa: E402
from utils import get_device, load_discriminator, load_generator, set_seed  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Only these training configs (e.g. baseline sn). Default: all.")
    parser.add_argument("--Ks", type=float, nargs="*", default=[2, 5, 10, 20])
    parser.add_argument("--psis", type=float, nargs="*", default=[2.0, 1.5, 1.0, 0.75, 0.5])
    parser.add_argument("--features", default="platform", choices=["platform", "mnist"],
                        help="platform: Inception FID + VGG-16 P/R (as the course platform); "
                             "mnist: cheap MNIST CNN features.")
    parser.add_argument("--out", default=None,
                        help="Default: results/<features>/sampling_metrics.csv")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--eval_seed", type=int, default=0)
    args = parser.parse_args()
    args.out = args.out or os.path.join("results", args.features, "sampling_metrics.csv")

    device = get_device()
    evaluator = Evaluator(device, data_path=os.getenv("DATA"), features=args.features)
    done = {(r["run"], r["sampler"], r["D_variant"], float(r["K"]), r["psi"])
            for r in read_rows(args.out)}

    for run_dir in list_runs(args.runs_dir):
        info = run_info(run_dir)
        if args.configs and info["config"] not in args.configs:
            continue
        ckpt = os.path.join(run_dir, "checkpoints")
        G = load_generator(ckpt, device)

        # (sampler, D file, K, psi, rescale)
        jobs = [("vanilla", None, 1.0, None, False), ("vanilla_rescale", None, 1.0, None, True)]
        jobs += [("truncation", None, 1.0, psi, False) for psi in args.psis]
        for d_file in ("D.pth", "D_calibrated.pth"):
            if os.path.exists(os.path.join(ckpt, d_file)):
                jobs += [(m, d_file, K, None, False) for m in ("drs", "obrs") for K in args.Ks]

        for sampler, d_file, K, psi, rescale in jobs:
            d_variant = {None: "none", "D.pth": "final", "D_calibrated.pth": "calibrated"}[d_file]
            key = (info["run"], sampler, d_variant, float(K), "" if psi is None else str(psi))
            if key in done:
                continue
            D = load_discriminator(ckpt, device, d_file) if d_file else None
            method = "vanilla" if sampler == "vanilla_rescale" else sampler
            set_seed(args.eval_seed)
            with torch.no_grad():
                images, stats = sample(G, args.n, device, method=method, D=D, K=K, psi=psi,
                                       rescale=rescale)
            metrics = evaluator(images)
            row = {**info, "sampler": sampler, "D_variant": d_variant, "K": K,
                   "psi": "" if psi is None else psi,
                   **{k: round(v, 5) for k, v in metrics.items()},
                   "acceptance_rate": round(stats["acceptance_rate"], 5),
                   "n_proposed": stats["n_proposed"], "n_calib": stats["n_calib"],
                   "time_s": round(stats["time_s"], 2)}
            append_row(args.out, row)
            print(row)


if __name__ == "__main__":
    main()
