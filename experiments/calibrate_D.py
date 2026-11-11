"""Ablation "quality of the density-ratio estimate": recalibrate D for a frozen G.

Rejection sampling assumes that D is the OPTIMAL discriminator for the final
generator, so that exp(logit D(x)) = p_data(x) / p_G(x). At the end of GAN
training, D is only approximately optimal (it was chasing a moving G). Here we
freeze G and keep training D alone for a few epochs, which brings it closer to
the optimal discriminator of this G (as done in the DRS paper).

    python experiments/calibrate_D.py --run_dir runs/sn_seed0 --epochs 3

Writes runs/sn_seed0/checkpoints/D_calibrated.pth. The generator is not modified.

--from_scratch: instead of fine-tuning the final D, train a NEW discriminator from a
random initialisation against the frozen G (same starter architecture, no spectral
normalisation, BCE loss). This isolates the quality of the density-ratio estimate
from GAN training: with the BCE loss, the optimal D satisfies
logit D(x) = log p_data(x) / p_G(x), which is exactly what rejection sampling needs,
whereas a D trained with the hinge loss (or bounded by SN) does not estimate this
ratio. Early stopping on held-out data (MNIST test set + fresh fakes) keeps the
epoch with the lowest validation loss, so D does not memorise the training digits.

    python experiments/calibrate_D.py --run_dir runs/sn_seed0 --from_scratch

Writes runs/sn_seed0/checkpoints/D_scratch.pth and D_scratch_log.csv.
"""
import argparse
import os
import time

from common import is_finished, run_info

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402

from model import Discriminator  # noqa: E402
from utils import (LATENT_DIM, MNIST_DIM, get_device, load_discriminator,  # noqa: E402
                   load_generator, set_seed)


def bce_loss(D, G, x_real, device, x_fake=None):
    """BCE loss of D on a batch of real images and as many fakes (fresh ones by default),
    and its accuracy."""
    if x_fake is None:
        with torch.no_grad():
            x_fake = G(torch.randn(len(x_real), LATENT_DIM, device=device))
    logit_real, logit_fake = D.logits(x_real), D.logits(x_fake)
    loss = (F.binary_cross_entropy_with_logits(logit_real, torch.ones_like(logit_real))
            + F.binary_cross_entropy_with_logits(logit_fake, torch.zeros_like(logit_fake)))
    acc = ((logit_real > 0).float().mean() + (logit_fake < 0).float().mean()) / 2
    return loss, acc.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Default: 3 (fine-tuning of the final D), 30 (--from_scratch).")
    parser.add_argument("--from_scratch", action="store_true",
                        help="Train a new D from a random initialisation (writes D_scratch.pth).")
    parser.add_argument("--patience", type=int, default=5,
                        help="--from_scratch: stop after this many epochs without a better "
                             "validation loss.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Recalibrate even if done already.")
    args = parser.parse_args()

    if not is_finished(args.run_dir):
        print(f"Skipping {args.run_dir}: training not finished.")
        return
    out_name = "D_scratch.pth" if args.from_scratch else "D_calibrated.pth"
    args.epochs = args.epochs or (30 if args.from_scratch else 3)
    if os.path.exists(os.path.join(args.run_dir, "checkpoints", out_name)) and not args.force:
        print(f"Skipping {args.run_dir}: {out_name} already exists (use --force to redo).")
        return
    set_seed(args.seed)
    device = get_device()
    ckpt = os.path.join(args.run_dir, "checkpoints")
    G = load_generator(ckpt, device)               # frozen: eval mode, never updated
    if args.from_scratch:
        D = Discriminator(MNIST_DIM, spectral_norm=False).to(device).train()   # random init
    else:
        D = load_discriminator(ckpt, device).train()
    opt = torch.optim.Adam(D.parameters(), lr=args.lr)

    data_path = os.getenv("DATA")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=data_path or "data", train=True, transform=transform,
                       download=data_path is None),
        batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(       # held-out real digits (test set)
        datasets.MNIST(root=data_path or "data", train=False, transform=transform,
                       download=data_path is None),
        batch_size=1000, shuffle=False, num_workers=4)

    with torch.no_grad():                           # the same 10 000 held-out fakes at every epoch
        val_fake = [G(torch.randn(1000, LATENT_DIM, device=device)) for _ in val_loader]

    def validate():
        D.eval()
        losses, accs = [], []
        with torch.no_grad():
            for (x, _), x_fake in zip(val_loader, val_fake):
                x_real = x.view(-1, MNIST_DIM).to(device)
                loss, acc = bce_loss(D, G, x_real, device, x_fake[:len(x_real)])
                losses.append(loss.item())
                accs.append(acc)
        D.train()
        return sum(losses) / len(losses), sum(accs) / len(accs)

    t0 = time.time()
    log, best, best_state, bad = [], float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        total, n = 0.0, 0
        for x, _ in loader:
            loss, _ = bce_loss(D, G, x.view(-1, MNIST_DIM).to(device), device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n = total + loss.item(), n + 1
        val_loss, val_acc = validate()
        log.append(f"{epoch},{total / n:.5f},{val_loss:.5f},{val_acc:.5f},{time.time() - t0:.1f}")
        print(f"{run_info(args.run_dir)['run']} | {out_name} epoch {epoch} | train loss "
              f"{total / n:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.4f}")
        if not args.from_scratch:
            continue
        if val_loss < best:
            best, bad = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in D.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping: best validation loss {best:.4f}")
                break

    if best_state is not None:
        D.load_state_dict(best_state)
    torch.save(D.state_dict(), os.path.join(ckpt, out_name))
    with open(os.path.join(ckpt, out_name.replace(".pth", "_log.csv")), "w") as f:
        f.write("epoch,train_loss,val_loss,val_acc,time_s\n" + "\n".join(log) + "\n")
    print(f"Saved {ckpt}/{out_name} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
