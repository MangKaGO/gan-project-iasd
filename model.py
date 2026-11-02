import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as _sn


class Generator(nn.Module):
    # The generator architecture is FIXED by the assignment: do not modify it.
    def __init__(self, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class Discriminator(nn.Module):
    """Same layers as the starter discriminator.

    Two additions, both optional and backward compatible:
      - spectral_norm=True wraps every linear layer with spectral normalisation
        (Miyato et al., 2018): each weight matrix is divided by its largest
        singular value, which bounds the Lipschitz constant of D.
      - logits(x) returns the score BEFORE the sigmoid. It is used by the
        rejection samplers, since exp(logit) estimates the density ratio
        p_data(x) / p_G(x) when D is close to optimal.
    With spectral_norm=False, the network is exactly the starter one, so the
    baseline is unchanged and old checkpoints still load.
    """

    def __init__(self, d_input_dim, spectral_norm=False):
        super(Discriminator, self).__init__()
        wrap = _sn if spectral_norm else (lambda layer: layer)
        self.fc1 = wrap(nn.Linear(d_input_dim, 1024))
        self.fc2 = wrap(nn.Linear(1024, 512))
        self.fc3 = wrap(nn.Linear(512, 256))
        self.fc4 = wrap(nn.Linear(256, 1))

    def logits(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return self.fc4(x)

    # forward method (unchanged behaviour: probability that x is real)
    def forward(self, x):
        return torch.sigmoid(self.logits(x))
