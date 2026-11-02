"""Generate exactly 10 000 MNIST images (28x28 PNG) in samples/.

Platform contract (unchanged):
    python generate.py --batch_size 256
loads the generator from checkpoints/ and writes samples/0.png ... samples/9999.png.

The sampling method is read from checkpoints/config.json (key "sampling"), so the
evaluation platform uses the method we choose without any extra argument, e.g.
    "sampling": {"method": "drs", "K": 5}
Without that key, the starter behaviour (vanilla sampling) is used.
Any value can be overridden on the command line (--method, --K, --psi, --rescale).
"""
import argparse
import os

import torch
import torchvision

from sampling import sample
from utils import get_device, load_config, load_discriminator, load_generator, set_seed

N_SAMPLES = 10000

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate MNIST samples with the trained GAN.')
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for generation.")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--out_dir", type=str, default="samples")
    parser.add_argument("--method", type=str, default=None,
                        choices=["vanilla", "truncation", "drs", "obrs"])
    parser.add_argument("--K", type=float, default=None, help="Sampling budget for drs / obrs.")
    parser.add_argument("--psi", type=float, default=None, help="Truncation threshold.")
    parser.add_argument("--rescale", action="store_true", default=None,
                        help="Save (x+1)/2 instead of clamping x to [0, 1].")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    set_seed(args.seed)

    # Command-line values override the ones stored in checkpoints/config.json.
    cfg = load_config(args.ckpt_dir).get("sampling", {})
    method = args.method or cfg.get("method", "vanilla")
    K = args.K if args.K is not None else cfg.get("K", 1.0)
    psi = args.psi if args.psi is not None else cfg.get("psi")
    rescale = args.rescale if args.rescale is not None else cfg.get("rescale", False)
    print(f"Sampling method: {method} (K={K}, psi={psi}, rescale={rescale})")

    print('Model Loading...')
    G = load_generator(args.ckpt_dir, device)
    D = load_discriminator(args.ckpt_dir, device) if method in ("drs", "obrs") else None
    print('Model loaded.')

    print('Start Generating')
    images, stats = sample(G, N_SAMPLES, device, method=method, D=D, K=K, psi=psi,
                           batch_size=args.batch_size, rescale=rescale)
    os.makedirs(args.out_dir, exist_ok=True)
    for k in range(N_SAMPLES):
        torchvision.utils.save_image(images[k], os.path.join(args.out_dir, f'{k}.png'))
    print(f"Saved {N_SAMPLES} images to {args.out_dir}/ | "
          f"acceptance rate {stats['acceptance_rate']:.3f} | time {stats['time_s']:.1f}s")
