import numpy as np
import torch
from torch import nn


class NN(nn.Module):  # [x,y]->[RGB]
    def __init__(self, dim_in, hidden_size, num_layers):
        super(NN, self).__init__()

        layers = [nn.Linear(dim_in, hidden_size), nn.ReLU()]
        for i in range(1, num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, 3))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


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
        b = 1 / self.in_f if self.is_first \
            else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


def gon_model(input_dim, hidden_dim, num_layers):
    layers = [SirenLayer(input_dim, hidden_dim, is_first=True)]
    for i in range(1, num_layers - 1):
        layers.append(SirenLayer(hidden_dim, hidden_dim))
    layers.append(SirenLayer(hidden_dim, 3, is_last=True))

    return nn.Sequential(*layers)


class NN_grownet(nn.Module):  # [x,y]->[RGB]
    def __init__(self, dim_in, hidden_size, num_layers):
        super(NN_grownet, self).__init__()

        layers = [nn.Linear(dim_in, hidden_size), nn.ReLU()]
        for i in range(1, num_layers - 1):
            layers.append([nn.Linear(hidden_size, hidden_size), nn.ReLU()])

        self.model_part1 = nn.Sequential(*layers)
        self.model_part2 = nn.Sequential(
            nn.Linear(hidden_size, 3), nn.Sigmoid()
        )

    def forward(self, x, prev_penultimate=None):
        if prev_penultimate is not None:
            x = torch.cat([x, prev_penultimate], dim=1)
        penultimate = self.model_part1(x)
        out = self.model_part2(penultimate)
        return penultimate, out

    @classmethod
    def get_model(cls, stage, input_size, penultimate_size, num_layers):
        if stage == 0:
            dim_in = input_size
        else:
            dim_in = input_size + penultimate_size
        model = NN_grownet(dim_in, penultimate_size, num_layers)
        return model


class SIREN_grownet(nn.Module):
    def __init__(self, dim_in, hidden_size, num_layers):
        super(SIREN_grownet, self).__init__()
        layers = [SirenLayer(dim_in, hidden_size, is_first=True)]
        for i in range(1, num_layers - 1):
            layers.append(SirenLayer(hidden_size, hidden_size))

        self.model_part1 = nn.Sequential(*layers)
        self.model_part2 = SirenLayer(hidden_size, 3, is_last=True)

    def forward(self, x, prev_penultimate=None):
        if prev_penultimate is not None:
            x = torch.cat([x, prev_penultimate], dim=1)
        penultimate = self.model_part1(x)
        out = self.model_part2(penultimate)
        return penultimate, out

    @classmethod
    def get_model(cls, stage, input_size, penultimate_size, num_layers):
        if stage == 0:
            dim_in = input_size
        else:
            dim_in = input_size + penultimate_size
        model = SIREN_grownet(dim_in, penultimate_size, num_layers)
        return model
