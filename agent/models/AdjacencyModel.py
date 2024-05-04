import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from networks.dreamer.utils import build_model
from networks.transformer.layers import AttentionEncoder


class AdjacencyModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, agents_num, activation=nn.ReLU, norm=nn.Identity):
        super().__init__()
        if activation == nn.ReLU:
            self.activation = 'relu'
        elif activation == nn.SiLU:
            self.activation = 'silu'
        else:
            raise NotImplementedError

        self.value_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self._att_net = AttentionEncoder(1, hidden_dim, hidden_dim, norm=norm)
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            norm(hidden_dim)
        )
        self.feedforward_model = build_model(hidden_dim, agents_num ** 2, 1, hidden_dim, activation, norm=norm)

    def forward(self, state_features):
        seq_length, batch_size = state_features.shape[0], state_features.shape[1]
        n_agents = state_features.shape[-2]
        if self.activation == 'relu':
            embeds = F.relu(self.embed(state_features))
        elif self.activation == 'silu':
            embeds = F.silu(self.embed(state_features))
        else:
            raise NotImplementedError
        embeds = embeds.view(-1, n_agents, embeds.shape[-1])
        h = self.value_token.repeat(seq_length * batch_size, 1, 1).to(state_features.device)
        embeds = torch.cat([h, embeds], dim=1)
        if self.activation == 'relu':
            attn_embeds = F.relu(self._att_net(embeds).view(seq_length * batch_size, n_agents, embeds.shape[-1]))
        elif self.activation == 'silu':
            attn_embeds = F.silu(self._att_net(embeds).view(seq_length * batch_size, n_agents, embeds.shape[-1]))
        else:
            raise NotImplementedError

        features = self.feedforward_model(attn_embeds)[0]
        matrix = td.independent.Independent(td.Bernoulli(logits=features), 1)

        return matrix
