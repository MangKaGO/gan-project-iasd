"""Evaluation of 10 000 generated images against 10 000 real MNIST test images.

Two feature spaces
------------------
features="platform"  (default) reproduces the course evaluation platform:
    FID        Inception-v3 pool features (2048-d, the standard pytorch-fid
               weights), computed on all 10 000 images.
    P / R      VGG-16 fc2 features (4096-d, torchvision ImageNet weights),
               k = 5, on 9 984 images (39 complete batches of 256).
    The weights are downloaded once (run on the login node, which has internet):
        python -c "from evaluation.metrics import download_weights; download_weights()"
    Exact implementation details of the platform (preprocessing, weights) may
    still differ slightly, so values are close to, not identical with, the
    leaderboard.

features="mnist"  a small CNN trained on MNIST (128-d, evaluation/mnist_cnn.pth),
    used for every metric, k = 3. Much cheaper, useful for quick checks.

Every run is evaluated with the same protocol, so comparisons are valid within
one feature space. Never mix results of the two feature spaces.

Metrics
-------
FID        Frechet distance between Gaussians fitted to real / fake features.
precision  Kynkaanniemi et al. (2019): fraction of generated samples inside the
           k-NN ball of at least one real sample (quality).
recall     Fraction of real samples inside the k-NN ball of at least one
           generated sample (diversity).
density    Naeem et al. (2020): like precision, but counts HOW MANY real balls
coverage   contain each fake (density) and the fraction of real balls that
           contain at least one fake (coverage). Both use only real-sample balls,
           so they are robust to outliers, unlike precision / recall.
classes    The MNIST CNN predicts the digit of each generated image:
           class_entropy = entropy of the predicted class histogram / log(10)
           (1 = balanced), min_class_frac = share of the rarest digit.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_PATH = os.path.join(HERE, "mnist_cnn.pth")
PLATFORM_K, PLATFORM_PR_N = 5, 9984          # k-NN size and number of images for P / R


# ----------------------------------------------------------------------------
# MNIST CNN (class statistics, and the cheap "mnist" feature space)
# ----------------------------------------------------------------------------

class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def features(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.dropout(torch.flatten(x, 1))
        return F.relu(self.fc1(x))

    def forward(self, x):
        return self.fc2(self.dropout(self.features(x)))


def _mnist(data_path, train):
    to_download = data_path is None
    return datasets.MNIST(root=data_path or "data", train=train,
                          transform=transforms.ToTensor(), download=to_download)


def get_feature_extractor(device, data_path=None, epochs=3):
    """Load the MNIST CNN, or train it once (about 1 min on GPU) if missing."""
    model = MnistCNN().to(device)
    if os.path.exists(CLASSIFIER_PATH):
        model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
        return model.eval()

    print("Training the MNIST classifier (done only once)...")
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(_mnist(data_path, True), batch_size=128, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            F.cross_entropy(model(x.to(device)), y.to(device)).backward()
            opt.step()
    model.eval()
    real, labels = load_real_images(data_path, with_labels=True)
    acc = (predict(model, real, device) == labels).float().mean().item()
    print(f"MNIST classifier test accuracy: {acc:.4f}")
    torch.save(model.state_dict(), CLASSIFIER_PATH)
    return model


def load_real_images(data_path=None, n=10000, with_labels=False):
    """The 10 000 MNIST test images, in [0, 1], shape (n, 1, 28, 28)."""
    ds = _mnist(data_path, False)
    x = ds.data[:n].float().div(255).unsqueeze(1)
    return (x, ds.targets[:n]) if with_labels else x


@torch.no_grad()
def _run(fn, images, device, batch_size=64):
    return torch.cat([fn(images[i:i + batch_size].to(device)).float().cpu()
                      for i in range(0, len(images), batch_size)])


def predict(model, images, device):
    return _run(lambda x: model(x).argmax(1), images, device).long()


def quantize(images):
    """Round to 8 bits, exactly as when the images are saved as PNG files."""
    return (images.clamp(0, 1) * 255).round() / 255


# ----------------------------------------------------------------------------
# ImageNet networks (the "platform" feature space)
# ----------------------------------------------------------------------------

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _inception(device):
    from pytorch_fid.inception import InceptionV3
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    # pytorch-fid takes 3-channel images in [0, 1], resizes to 299 and normalises itself.
    return lambda x: model(x.repeat(1, 3, 1, 1))[0].flatten(1)


def _vgg16_fc2(device):
    from torchvision.models import vgg16
    weights = None if os.getenv("EVAL_RANDOM_WEIGHTS") else "IMAGENET1K_V1"   # random: tests only
    model = vgg16(weights=weights).to(device).eval()
    head = model.classifier[:5]          # fc1, ReLU, Dropout, fc2, ReLU  ->  4096-d
    mean, std = _IMAGENET_MEAN.to(device), _IMAGENET_STD.to(device)

    def fn(x):
        x = F.interpolate(x.repeat(1, 3, 1, 1), size=(224, 224), mode="bilinear", align_corners=False)
        x = model.avgpool(model.features((x - mean) / std))
        return head(torch.flatten(x, 1))
    return fn


def download_weights():
    """Download Inception and VGG-16 weights into the torch cache (needs internet)."""
    dev = torch.device("cpu")
    _inception(dev)
    _vgg16_fc2(dev)
    print("Inception and VGG-16 weights are in the torch cache.")


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def fid(f_real, f_fake):
    f_real, f_fake = f_real.double().numpy(), f_fake.double().numpy()
    mu1, mu2 = f_real.mean(0), f_fake.mean(0)
    s1, s2 = np.cov(f_real, rowvar=False), np.cov(f_fake, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)              # works with every scipy version
    if not np.isfinite(covmean).all():          # numerical safety
        eps = 1e-6 * np.eye(len(s1))
        covmean = linalg.sqrtm((s1 + eps) @ (s2 + eps))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def _knn_radii(f, k, device, chunk=1000):
    """Distance from each point of f to its k-th nearest neighbour in f."""
    f = f.to(device)
    radii = []
    for i in range(0, len(f), chunk):
        d = torch.cdist(f[i:i + chunk], f)
        # k + 1 because the closest point is the point itself (distance 0).
        radii.append(d.kthvalue(k + 1, dim=1).values)
    return torch.cat(radii)


def _balls(f_query, f_ref, radii_ref, device, chunk=1000):
    """For each query point, the number of balls B(ref_j, radius_j) containing
    it; and for each ball, whether it contains at least one query point."""
    f_ref, radii_ref = f_ref.to(device), radii_ref.to(device)
    counts, hit = [], torch.zeros(len(f_ref), dtype=torch.bool, device=device)
    for i in range(0, len(f_query), chunk):
        inside = torch.cdist(f_query[i:i + chunk].to(device), f_ref) <= radii_ref[None, :]
        counts.append(inside.sum(dim=1).cpu())
        hit |= inside.any(dim=0)
    return torch.cat(counts).float(), hit.float().cpu()


def prdc(f_real, f_fake, device, k):
    """Precision, recall (Kynkaanniemi et al.), density, coverage (Naeem et al.)."""
    r_real, r_fake = _knn_radii(f_real, k, device), _knn_radii(f_fake, k, device)
    fake_in_real, real_balls_hit = _balls(f_fake, f_real, r_real, device)
    real_in_fake, _ = _balls(f_real, f_fake, r_fake, device)
    return {"precision": (fake_in_real > 0).float().mean().item(),
            "recall": (real_in_fake > 0).float().mean().item(),
            "density": (fake_in_real.mean() / k).item(),
            "coverage": real_balls_hit.mean().item()}


class Evaluator:
    """Compute all metrics for a batch of generated images.

    The real features are computed once and reused for every evaluation."""

    def __init__(self, device, data_path=None, features="platform"):
        self.device, self.features = device, features
        self.classifier = get_feature_extractor(device, data_path)
        real = load_real_images(data_path)
        if features == "platform":
            self.k, self.n_pr = PLATFORM_K, PLATFORM_PR_N
            self.f_fid, self.f_pr = _inception(device), _vgg16_fc2(device)
        elif features == "mnist":
            self.k, self.n_pr = 3, len(real)
            self.f_fid = self.f_pr = self.classifier.features
        else:
            raise ValueError(features)
        self.real_fid = _run(self.f_fid, real, device)
        self.real_pr = _run(self.f_pr, real[:self.n_pr], device)

    def __call__(self, images):
        images = quantize(images)
        out = {"fid": fid(self.real_fid, _run(self.f_fid, images, self.device))}
        out.update(prdc(self.real_pr, _run(self.f_pr, images[:self.n_pr], self.device),
                        self.device, self.k))
        p = torch.bincount(predict(self.classifier, images, self.device), minlength=10).float()
        p = p / p.sum()
        out["class_entropy"] = abs((p[p > 0] * p[p > 0].log()).sum().item()) / np.log(10)
        out["min_class_frac"] = p.min().item()
        return out
