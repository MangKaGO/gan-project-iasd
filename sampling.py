"""Sampling methods applied to a TRAINED generator (inference lever).

All functions return images as a tensor of shape (n, 1, 28, 28) with values in
[0, 1], plus a dict of statistics (acceptance rate, number of proposals, time).

Methods
-------
vanilla     z ~ N(0, I), x = G(z).                         (starter behaviour)
truncation  z ~ N(0, I) restricted to |z_i| <= psi (BigGAN truncation trick).
drs         Discriminator Rejection Sampling (Azadi et al., 2019), course def. 3.1:
                a(x) = r(x) / ( r(x) (1 - e^gamma) + M e^gamma )
obrs        Optimal Budgeted Rejection Sampling (Verine et al., 2024), theorem 3.2:
                a(x) = min( r(x) c_K / M , 1 ),  with c_K such that E[a] = 1/K.

Both rejection samplers use r(x) = exp(logit D(x)), which estimates the density
ratio p_data(x) / p_G(x) when D is close to the optimal discriminator, and
M = max r(x), estimated on a calibration batch of fresh samples.

Budget
------
For DRS and OBRS we fix the budget K: on average K candidates are generated per
kept image (acceptance rate 1/K). For DRS, gamma is chosen so that its acceptance
rate is also 1/K. Both samplers are therefore compared at the same sampling cost.
"""
import time

import torch
import torch.nn.functional as F

from utils import LATENT_DIM, MNIST_DIM


# ----------------------------------------------------------------------------
# Latent sampling and image conversion
# ----------------------------------------------------------------------------

def sample_latent(n, device, psi=None):
    """z ~ N(0, I). With psi, every coordinate with |z_i| > psi is resampled
    until it falls inside [-psi, psi] (truncated normal)."""
    z = torch.randn(n, LATENT_DIM, device=device)
    if psi is not None:
        outside = z.abs() > psi
        while outside.any():
            z[outside] = torch.randn(int(outside.sum()), device=device)
            outside = z.abs() > psi
    return z


def to_image(x, rescale=False):
    """Convert generator outputs in [-1, 1] to images in [0, 1].

    rescale=False reproduces the starter generate.py exactly: save_image clamps
    values to [0, 1], so every negative pixel becomes black.
    rescale=True applies the exact inverse of the training normalisation,
    (x + 1) / 2.
    """
    x = x.view(-1, 1, 28, 28)
    return ((x + 1) / 2 if rescale else x).clamp(0, 1)


# ----------------------------------------------------------------------------
# Acceptance functions (log-space for numerical stability)
# ----------------------------------------------------------------------------

def log_accept_drs(log_r, log_M, gamma):
    """log a(x) for DRS. Rewriting the course formula:
        a = 1 / (1 + e^gamma (M / r - 1))
    so log a = -softplus(gamma + log(M / r - 1))."""
    d = (log_M - log_r).clamp(min=1e-6)        # log(M / r) >= 0 (r > M is clamped)
    return -F.softplus(gamma + torch.log(torch.expm1(d)))


def log_accept_obrs(log_r, log_M, log_c):
    """log a(x) for OBRS: log min(r c / M, 1)."""
    return (log_r + log_c - log_M).clamp(max=0.0)


def _bisect(f, lo, hi, target, increasing, iters=100):
    """Solve f(t) = target for a monotone function f on [lo, hi]."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        too_high = f(mid) > target
        if too_high == increasing:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def calibrate(log_r_calib, method, K):
    """Find the parameter (gamma for DRS, log c_K for OBRS) giving an average
    acceptance of 1/K on the calibration logits, and return log M."""
    log_M = log_r_calib.max()
    target = 1.0 / K
    if method == "drs":
        mean_acc = lambda g: log_accept_drs(log_r_calib, log_M, g).exp().mean().item()
        param = _bisect(mean_acc, -60.0, 60.0, target, increasing=False)
    elif method == "obrs":
        # Theorem 3.2 requires c_K >= 1 (log c_K >= 0). If even c_K = 1 accepts
        # more than 1/K, the budget K is larger than needed: OBRS is then plain
        # rejection sampling (a = r / M) and its acceptance rate stays above 1/K.
        mean_acc = lambda c: log_accept_obrs(log_r_calib, log_M, c).exp().mean().item()
        if mean_acc(0.0) >= target:
            print(f"OBRS: K={K} exceeds the useful budget, using c_K = 1 "
                  f"(acceptance {mean_acc(0.0):.3f} > 1/K).")
            param = 0.0
        else:
            param = _bisect(mean_acc, 0.0, 60.0, target, increasing=True)
    else:
        raise ValueError(method)
    return log_M, param


# ----------------------------------------------------------------------------
# Samplers
# ----------------------------------------------------------------------------

@torch.no_grad()
def sample(G, n, device, method="vanilla", D=None, K=1.0, psi=None,
           batch_size=2048, n_calib=10000, rescale=False):
    """Generate exactly n images with the chosen method.

    Returns (images in [0, 1] on CPU, stats dict)."""
    t0 = time.time()
    if method in ("vanilla", "truncation") or K <= 1:
        out, n_prop = [], 0
        while n_prop < n:
            b = min(batch_size, n - n_prop)
            z = sample_latent(b, device, psi if method == "truncation" else None)
            out.append(to_image(G(z), rescale).cpu())
            n_prop += b
        images = torch.cat(out)[:n]
        return images, {"method": method, "K": 1.0, "psi": psi, "n_proposed": n_prop,
                        "n_calib": 0, "acceptance_rate": 1.0, "time_s": time.time() - t0}

    if D is None:
        raise ValueError(f"{method} needs the discriminator D.")

    # 1) Calibration: estimate M and the acceptance parameter on fresh samples.
    calib = []
    for i in range(0, n_calib, batch_size):
        z = sample_latent(min(batch_size, n_calib - i), device)
        calib.append(D.logits(G(z)).view(-1))
    log_r_calib = torch.cat(calib)
    log_M, param = calibrate(log_r_calib, method, K)

    # 2) Rejection loop until n images are accepted.
    accepted, n_acc, n_prop = [], 0, 0
    while n_acc < n:
        z = sample_latent(batch_size, device)
        x = G(z)
        log_r = D.logits(x).view(-1)
        if method == "drs":
            log_a = log_accept_drs(log_r, log_M, param)
        else:
            log_a = log_accept_obrs(log_r, log_M, param)
        keep = torch.rand_like(log_a).log() < log_a
        accepted.append(to_image(x[keep], rescale).cpu())
        n_acc += int(keep.sum())
        n_prop += batch_size

    images = torch.cat(accepted)[:n]
    return images, {"method": method, "K": float(K), "psi": None, "n_proposed": n_prop,
                    "n_calib": n_calib, "acceptance_rate": n_acc / n_prop,
                    "param": float(param), "time_s": time.time() - t0}
