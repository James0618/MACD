import torch.nn as nn
import torch.nn.functional as F

from networks.dreamer.utils import build_model


class Decoder(nn.Module):
    def __init__(self, embed, hidden, out_dim, layers=2, activation=nn.ReLU, norm=nn.Identity):
        super().__init__()
        if activation == nn.ReLU:
            self.activation = 'relu'
        elif activation == nn.SiLU:
            self.activation = 'silu'
        else:
            raise NotImplementedError

        self.fc1 = build_model(embed, hidden, layers, hidden, activation, norm=norm)
        self.fc2 = nn.Sequential(
            nn.Linear(hidden, out_dim),
            norm(out_dim),
        )

    def forward(self, z):
        if self.activation == 'relu':
            x = F.relu(self.fc1(z))
        elif self.activation == 'silu':
            x = F.silu(self.fc1(z))
        else:
            raise NotImplementedError

        return self.fc2(x), x


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden, embed, layers=2, activation=nn.ReLU, norm=nn.Identity):
        super().__init__()
        if activation == nn.ReLU:
            self.activation = 'relu'
        elif activation == nn.SiLU:
            self.activation = 'silu'
        else:
            raise NotImplementedError

        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hidden),
            norm(hidden),
            activation(),
        )
        self.encoder = build_model(hidden, embed, layers, hidden, activation, norm=norm)

    def forward(self, x):
        embed = self.fc1(x)
        if self.activation == 'relu':
            return self.encoder(F.relu(embed))
        elif self.activation == 'silu':
            return self.encoder(F.silu(embed))
        else:
            raise NotImplementedError
