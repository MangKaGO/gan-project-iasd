"""Ablation "quality of the density-ratio estimate": recalibrate D for a frozen G.

Rejection sampling assumes that D is the OPTIMAL discriminator for the final
generator, so that exp(logit D(x)) = p_data(x) / p_G(x). At the end of GAN
training, D is only approximately optimal (it was chasing a moving G). Here we
freeze G and keep training D alone for a few epochs, which brings it closer to
the optimal discriminator of this G (as done in the DRS paper).

    python experiments/calibrate_D.py --run_dir runs/sn_seed0 --epochs 3

Writes runs/sn_seed0/checkpoints/D_calibrated.pth. The generator is not modified.
"""
import argparse
import os
import time

from common import run_info

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402

from utils import (LATENT_DIM, MNIST_DIM, get_device, load_discriminator,  # noqa: E402
                   load_generator, set_seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    ckpt = os.path.join(args.run_dir, "checkpoints")
    G = load_generator(ckpt, device)               # frozen: eval mode, never updated
    D = load_discriminator(ckpt, device).train()
    opt = torch.optim.Adam(D.parameters(), lr=args.lr)

    data_path = os.getenv("DATA")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=data_path or "data", train=True, transform=transform,
                       download=data_path is None),
        batch_size=args.batch_size, shuffle=True, num_workers=4)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        total, n = 0.0, 0
        for x, _ in loader:
            x_real = x.view(-1, MNIST_DIM).to(device)
            with torch.no_grad():
                x_fake = G(torch.randn(len(x_real), LATENT_DIM, device=device))
            logit_real, logit_fake = D.logits(x_real), D.logits(x_fake)
            loss = (F.binary_cross_entropy_with_logits(logit_real, torch.ones_like(logit_real))
                    + F.binary_cross_entropy_with_logits(logit_fake, torch.zeros_like(logit_fake)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n = total + loss.item(), n + 1
        print(f"{run_info(args.run_dir)['run']} | calibration epoch {epoch} | D loss {total / n:.4f}")

    torch.save(D.state_dict(), os.path.join(ckpt, "D_calibrated.pth"))
    print(f"Saved {ckpt}/D_calibrated.pth ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
