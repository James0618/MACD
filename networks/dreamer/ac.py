import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.dreamer.utils import build_model
from networks.transformer.layers import AttentionEncoder
from torch.distributions import OneHotCategorical


class AttentionQNetwork(nn.Module):
    def __init__(self, in_dim, hidden_size, out_dim, activation=nn.ReLU):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden_size)
        self._attention_stack = AttentionEncoder(1, hidden_size, hidden_size)
        self.feedforward_model = build_model(hidden_size, out_dim, 1, hidden_size, activation)

    def forward(self, state_features):
        n_agents = state_features.shape[-2]
        batch_size = state_features.shape[:-2]
        embeds = F.relu(self.embed(state_features))
        embeds = embeds.view(-1, n_agents, embeds.shape[-1])
        attn_embeds = F.relu(self._attention_stack(embeds).view(*batch_size, n_agents, embeds.shape[-1]))

        q_values = self.feedforward_model(attn_embeds)

        return q_values

    def get_q_values(self, state_features, actions):
        q_values = self.forward(state_features)
        q_value = q_values.gather(-1, actions.argmax(-1, keepdim=True)).squeeze(-1)

        return q_value


class SACActor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, layers, activation=nn.ReLU, norm=nn.Identity):
        super().__init__()
        self.feedforward_model = build_model(in_dim, out_dim, layers, hidden_size, activation, norm=norm)

    def forward(self, state_features):
        x = self.feedforward_model(state_features)
        action_dist = OneHotCategorical(logits=x)
        action = action_dist.sample()

        return action, action_dist


class SoftActorCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pi = SACActor(config.FEAT, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS)
        self.q1 = AttentionQNetwork(config.FEAT, config.HIDDEN, config.ACTION_SIZE)
        self.q2 = AttentionQNetwork(config.FEAT, config.HIDDEN, config.ACTION_SIZE)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            action, action_dist = self.pi(obs)
            if deterministic:
                return action_dist.probs.argmax(dim=-1, keepdim=True)
            else:
                return action
