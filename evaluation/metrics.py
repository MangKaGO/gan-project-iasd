"""Evaluation of 10 000 generated images against 10 000 real MNIST test images.

Feature space
-------------
Inception-v3 features are designed for ImageNet photos, not for 28x28 digits.
We use instead the 128-d penultimate layer of a small CNN trained on MNIST
(evaluation/mnist_cnn.pth, trained once by get_feature_extractor). Absolute
values are therefore not comparable with papers or with the course platform,
but every method is evaluated with exactly the same protocol, so comparisons
between our runs are valid.

Metrics
-------
FID        Frechet distance between Gaussians fitted to real / fake features.
precision  Kynkaanniemi et al. (2019), k-NN estimator: fraction of generated
           samples that fall inside the k-NN ball of at least one real sample
           (quality).
recall     Fraction of real samples inside the k-NN ball of at least one
           generated sample (diversity / coverage).
classes    The classifier also predicts the digit of each generated image:
           class_entropy = entropy of the predicted class histogram divided by
           log(10) (1 = perfectly balanced), min_class_frac = rarest digit.
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


# ----------------------------------------------------------------------------
# Feature extractor
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

    print("Training the MNIST feature extractor (done only once)...")
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
    print(f"Feature extractor test accuracy: {acc:.4f}")
    torch.save(model.state_dict(), CLASSIFIER_PATH)
    return model


def load_real_images(data_path=None, n=10000, with_labels=False):
    """The 10 000 MNIST test images, in [0, 1], shape (n, 1, 28, 28)."""
    ds = _mnist(data_path, False)
    x = ds.data[:n].float().div(255).unsqueeze(1)
    return (x, ds.targets[:n]) if with_labels else x


@torch.no_grad()
def _run(model, images, device, fn, batch_size=1000):
    return torch.cat([fn(images[i:i + batch_size].to(device)).cpu()
                      for i in range(0, len(images), batch_size)])


def extract_features(model, images, device):
    return _run(model, images, device, model.features)


def predict(model, images, device):
    return _run(model, images, device, lambda x: model(x).argmax(1))


def quantize(images):
    """Round to 8 bits, exactly as when the images are saved as PNG files."""
    return (images.clamp(0, 1) * 255).round() / 255


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


def _coverage(f_query, f_ref, radii_ref, device, chunk=1000):
    """Fraction of query points inside at least one ball B(ref_j, radius_j)."""
    f_ref, radii_ref = f_ref.to(device), radii_ref.to(device)
    inside = []
    for i in range(0, len(f_query), chunk):
        d = torch.cdist(f_query[i:i + chunk].to(device), f_ref)
        inside.append((d <= radii_ref[None, :]).any(dim=1))
    return torch.cat(inside).float().mean().item()


def precision_recall(f_real, f_fake, device, k=3):
    precision = _coverage(f_fake, f_real, _knn_radii(f_real, k, device), device)
    recall = _coverage(f_real, f_fake, _knn_radii(f_fake, k, device), device)
    return precision, recall


class Evaluator:
    """Compute all metrics for a batch of generated images.

    The real features are computed once and reused for every evaluation."""

    def __init__(self, device, data_path=None, k=3):
        self.device, self.k = device, k
        self.model = get_feature_extractor(device, data_path)
        self.f_real = extract_features(self.model, load_real_images(data_path), device)

    def __call__(self, images):
        images = quantize(images)
        f_fake = extract_features(self.model, images, self.device)
        precision, recall = precision_recall(self.f_real, f_fake, self.device, self.k)
        counts = torch.bincount(predict(self.model, images, self.device), minlength=10).float()
        p = counts / counts.sum()
        entropy = abs((p[p > 0] * p[p > 0].log()).sum().item()) / np.log(10)
        return {"fid": fid(self.f_real, f_fake), "precision": precision, "recall": recall,
                "class_entropy": entropy, "min_class_frac": p.min().item()}
