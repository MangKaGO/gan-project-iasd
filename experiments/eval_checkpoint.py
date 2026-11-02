"""Evaluate ONE checkpoint folder, e.g. the one submitted to the platform.

    python experiments/eval_checkpoint.py --ckpt_dir checkpoints
    python experiments/eval_checkpoint.py --ckpt_dir runs/sn_seed0/checkpoints --method drs --K 5

Use it to check that our "platform" evaluation reproduces the leaderboard
(same generator -> FID, precision and recall should be close).
"""
import argparse
import os

import common  # noqa: F401  (makes the repository root importable)

import torch  # noqa: E402

from evaluation.metrics import Evaluator  # noqa: E402
from sampling import sample  # noqa: E402
from utils import get_device, load_discriminator, load_generator, set_seed  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--features", default="platform", choices=["platform", "mnist"])
    parser.add_argument("--method", default="vanilla", choices=["vanilla", "truncation", "drs", "obrs"])
    parser.add_argument("--K", type=float, default=1.0)
    parser.add_argument("--psi", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    G = load_generator(args.ckpt_dir, device)
    D = load_discriminator(args.ckpt_dir, device) if args.method in ("drs", "obrs") else None
    set_seed(args.seed)
    with torch.no_grad():
        images, stats = sample(G, 10000, device, method=args.method, D=D, K=args.K, psi=args.psi)
    metrics = Evaluator(device, data_path=os.getenv("DATA"), features=args.features)(images)
    print({k: round(float(v), 4) for k, v in metrics.items()},
          f"| acceptance {stats['acceptance_rate']:.3f} | generation {stats['time_s']:.1f}s")


if __name__ == "__main__":
    main()
