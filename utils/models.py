import numpy as np
import torch
from torch import nn


class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / \
            self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


def gon_model(num_layers, input_dim, hidden_dim):
    layers = [SirenLayer(input_dim, hidden_dim, is_first=True)]
    for i in range(1, num_layers - 1):
        layers.append(SirenLayer(hidden_dim, hidden_dim))
    layers.append(SirenLayer(hidden_dim, 3, is_last=True))

    return nn.Sequential(*layers)


class NN(nn.Module):  # in: x,y --> out: RGB
    def __init__(self, hidden_size):
        super(NN, self).__init__()
        self.model = nn.Sequential(
            # nn.Linear(2, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 3), nn.Sigmoid(),
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    # self.model = self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)
