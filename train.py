"""Train the GAN on MNIST.

Backward compatible with the starter code:
    python train.py --epochs 100 --lr 0.0002 --batch_size 64
still writes checkpoints/G.pth and checkpoints/D.pth.

New options used by the experiments:
    --seed 0                 fix all random generators
    --spectral_norm          spectral normalisation on D (method 1)
    --lr_d / --lr_g          separate learning rates (TTUR ablation)
    --run_dir runs/sn_seed0  save everything of this run in its own folder:
        runs/sn_seed0/config.json          all the arguments of the run
        runs/sn_seed0/log.csv              one line per epoch (losses, D scores, time)
        runs/sn_seed0/checkpoints/G_epoch010.pth, D_epoch010.pth, ...
        runs/sn_seed0/checkpoints/G.pth, D.pth   final weights
        runs/sn_seed0/grids/epoch_010.png  64 samples from a FIXED latent batch
"""
import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from model import Discriminator, Generator
from utils import (LATENT_DIM, MNIST_DIM, D_train, G_train, get_device,
                   save_config, save_models, set_seed)

# Same latent batch for every run, so grids of different methods are comparable.
FIXED_Z_SEED = 1234


def parse_args():
    parser = argparse.ArgumentParser(description='Train GAN on MNIST.')
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate (G and D).")
    parser.add_argument("--lr_g", type=float, default=None, help="Learning rate of G (default: --lr).")
    parser.add_argument("--lr_d", type=float, default=None, help="Learning rate of D (default: --lr).")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of mini-batches for SGD.")
    parser.add_argument("--gpus", type=int, default=-1, help="Number of GPUs to use (-1 for all available).")
    parser.add_argument("--spectral_norm", action="store_true", help="Spectral normalisation on D.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Folder for this run. Default: write to checkpoints/ like the starter code.")
    parser.add_argument("--save_every", type=int, default=10, help="Save a checkpoint every N epochs.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Random data and 2 batches per epoch, only to check that the code runs.")
    args = parser.parse_args()
    args.lr_g = args.lr_g or args.lr
    args.lr_d = args.lr_d or args.lr
    return args


def get_loader(args):
    if args.dry_run:
        x = torch.rand(2 * args.batch_size, 1, 28, 28) * 2 - 1
        dataset = torch.utils.data.TensorDataset(x, torch.zeros(len(x)))
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    data_path = os.getenv('DATA')
    to_download = data_path is None
    if to_download:
        data_path = "data"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_dataset = datasets.MNIST(root=data_path, train=True, transform=transform, download=to_download)
    return torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True)


def save_grid(G, z_fixed, path):
    G.eval()
    with torch.no_grad():
        x = G(z_fixed).view(-1, 1, 28, 28)
    G.train()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Images are in [-1, 1] (tanh): rescale to [0, 1] before saving.
    torchvision.utils.save_image((x + 1) / 2, path, nrow=8)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    if device.type == "cuda" and args.gpus == -1:
        args.gpus = torch.cuda.device_count()

    # Where to save. Without --run_dir we keep the starter behaviour.
    if args.run_dir is None:
        ckpt_dir, log_dir = "checkpoints", None
    else:
        ckpt_dir, log_dir = os.path.join(args.run_dir, "checkpoints"), args.run_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    config = {k: v for k, v in vars(args).items() if k != "dry_run"}
    save_config(config, ckpt_dir)           # read by generate.py (D architecture)
    if log_dir:
        save_config(config, log_dir)

    print('Dataset loading...')
    train_loader = get_loader(args)
    print('Dataset loaded.')

    G = Generator(g_output_dim=MNIST_DIM).to(device)
    D = Discriminator(MNIST_DIM, spectral_norm=args.spectral_norm).to(device)
    if args.gpus > 1:
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)

    criterion = nn.BCELoss()
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr_g)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr_d)

    z_fixed = torch.randn(64, LATENT_DIM, generator=torch.Generator().manual_seed(FIXED_Z_SEED)).to(device)

    log_rows = []
    print('Start training:')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        sums = {"loss_D": 0.0, "loss_G": 0.0, "D_real": 0.0, "D_fake": 0.0}
        n_batches = 0
        for x, _ in train_loader:
            x = x.view(-1, MNIST_DIM).to(device)
            loss_D, d_real, d_fake = D_train(x, G, D, D_optimizer, criterion, device)
            loss_G = G_train(x, G, D, G_optimizer, criterion, device)
            sums["loss_D"] += loss_D
            sums["loss_G"] += loss_G
            sums["D_real"] += d_real
            sums["D_fake"] += d_fake
            n_batches += 1

        row = {"epoch": epoch, **{k: v / n_batches for k, v in sums.items()},
               "epoch_time_s": time.time() - t0}
        log_rows.append(row)
        print(f"epoch {epoch:3d} | loss_D {row['loss_D']:.3f} | loss_G {row['loss_G']:.3f} | "
              f"D(real) {row['D_real']:.3f} | D(fake) {row['D_fake']:.3f} | {row['epoch_time_s']:.1f}s")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_models(G, D, ckpt_dir)                      # latest: G.pth / D.pth
            if log_dir:
                save_models(G, D, ckpt_dir, suffix=f"_epoch{epoch:03d}")
                save_grid(G, z_fixed, os.path.join(log_dir, "grids", f"epoch_{epoch:03d}.png"))
                with open(os.path.join(log_dir, "log.csv"), "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(log_rows)

    print('Training done.')


if __name__ == '__main__':
    main()
