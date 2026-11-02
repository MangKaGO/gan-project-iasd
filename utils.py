import json
import os
import random

import numpy as np
import torch

from model import Discriminator, Generator

MNIST_DIM = 784
LATENT_DIM = 100


def D_train(x, G, D, D_optimizer, criterion, device):
    #=======================Train the discriminator=======================#
    # Same update as the starter code. It also returns the mean scores
    # D(real) and D(fake), which we log to monitor training stability.
    D.zero_grad()

    # train discriminator on real
    x_real = x.to(device)
    y_real = torch.ones(x.shape[0], 1, device=device)
    D_real = D(x_real)
    D_real_loss = criterion(D_real, y_real)

    # train discriminator on fake
    z = torch.randn(x.shape[0], LATENT_DIM, device=device)
    x_fake = G(z)
    y_fake = torch.zeros(x.shape[0], 1, device=device)
    D_fake = D(x_fake)
    D_fake_loss = criterion(D_fake, y_fake)

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.item(), D_real.mean().item(), D_fake.mean().item()


def D_train_hinge(x, G, D, D_optimizer, device):
    """Hinge loss for D (Miyato et al., 2018; Lim & Ye, 2017), on the logits:
        L_D = E[max(0, 1 - D(x))] + E[max(0, 1 + D(G(z)))]
    Returns the loss and the mean sigmoid scores, logged like the BCE version."""
    D.zero_grad()
    logit_real = D.logits(x.to(device))
    z = torch.randn(x.shape[0], LATENT_DIM, device=device)
    logit_fake = D.logits(G(z))
    D_loss = torch.relu(1 - logit_real).mean() + torch.relu(1 + logit_fake).mean()
    D_loss.backward()
    D_optimizer.step()
    return D_loss.item(), torch.sigmoid(logit_real).mean().item(), torch.sigmoid(logit_fake).mean().item()


def G_train_hinge(x, G, D, G_optimizer, device):
    """Generator loss of the hinge GAN: L_G = -E[D(G(z))] (on the logits)."""
    G.zero_grad()
    z = torch.randn(x.shape[0], LATENT_DIM, device=device)
    G_loss = -D.logits(G(z)).mean()
    G_loss.backward()
    G_optimizer.step()
    return G_loss.item()


def G_train(x, G, D, G_optimizer, criterion, device):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], LATENT_DIM, device=device)
    y = torch.ones(x.shape[0], 1, device=device)

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.item()


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------

def set_seed(seed):
    """Fix every random number generator, so a run can be reproduced."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------------------------------------------------------
# Saving and loading
# ----------------------------------------------------------------------------

def _unwrap(model):
    """Remove the DataParallel wrapper, if any."""
    return model.module if hasattr(model, "module") else model


def _strip_module(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def save_models(G, D, folder, suffix=""):
    """Save G and D as G{suffix}.pth and D{suffix}.pth (suffix e.g. '_epoch050')."""
    os.makedirs(folder, exist_ok=True)
    torch.save(_unwrap(G).state_dict(), os.path.join(folder, f"G{suffix}.pth"))
    torch.save(_unwrap(D).state_dict(), os.path.join(folder, f"D{suffix}.pth"))


def save_config(config, folder):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def load_config(folder):
    """Read folder/config.json, or return {} if it does not exist."""
    path = os.path.join(folder, "config.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_model(G, folder, device, filename="G.pth"):
    """Load the generator weights (same behaviour as the starter code)."""
    ckpt = torch.load(os.path.join(folder, filename), map_location=device)
    G.load_state_dict(_strip_module(ckpt))
    return G


def load_generator(folder, device, filename="G.pth"):
    G = Generator(g_output_dim=MNIST_DIM).to(device)
    G = load_model(G, folder, device, filename)
    return G.eval()


def load_discriminator(folder, device, filename="D.pth", spectral_norm=None):
    """Load D. If spectral_norm is None, it is read from folder/config.json,
    because a spectrally normalised D has different parameter names."""
    if spectral_norm is None:
        spectral_norm = load_config(folder).get("spectral_norm", False)
    D = Discriminator(MNIST_DIM, spectral_norm=spectral_norm).to(device)
    ckpt = torch.load(os.path.join(folder, filename), map_location=device)
    D.load_state_dict(_strip_module(ckpt))
    return D.eval()
