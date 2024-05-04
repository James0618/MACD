import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import OneHotCategorical
from networks.transformer.layers import AttentionEncoder
from networks.dreamer.utils import build_model
from networks.transformer.transformer_layers import TransformerBlock


class Actor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, layers, activation=nn.ReLU, norm=nn.Identity):
        super().__init__()
        self.feedforward_model = build_model(in_dim, out_dim, layers, hidden_size, activation, norm=norm)

    def forward(self, state_features):
        x = self.feedforward_model(state_features)
        action_dist = OneHotCategorical(logits=x)
        action = action_dist.sample()

        return action, x


class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class ContinuousActor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, layers, activation=nn.ReLU, norm=nn.Identity):
        super().__init__()
        self.feedforward_model = build_model(
            in_dim, out_dim, layers, hidden_size, activation, norm=norm, if_last_norm=False
        )
        self.log_std = nn.Parameter(torch.ones(out_dim))

    def forward(self, state_features, deterministic=False):
        x_mean = self.feedforward_model(state_features)
        # reshape the action mean to be in [-1, 1]
        # x_mean = torch.tanh(x_mean) * 2
        x_std = (torch.sigmoid(self.log_std) * 0.5).to(state_features.device)
        action_dist = FixedNormal(x_mean, x_std)
        if deterministic:
            action = x_mean
        else:
            action = action_dist.sample()

        # action = torch.clamp(action, -1, 1)
        action_log_probs = action_dist.log_probs(action)

        return action, action_log_probs, action_dist.mean, action_dist.stddev

    def get_probs(self, state_features, actions):
        x_mean = self.feedforward_model(state_features)
        x_std = (torch.sigmoid(self.log_std) * 0.5).to(state_features.device)
        action_dist = FixedNormal(x_mean, x_std)
        action_log_probs = action_dist.log_probs(actions)

        return action_log_probs

    def evaluate_actions(self, state_features, actions):
        x_mean = self.feedforward_model(state_features)
        x_std = (torch.sigmoid(self.log_std) * 0.5).to(state_features.device)
        action_dist = FixedNormal(x_mean, x_std)
        action_log_probs = action_dist.log_probs(actions)
        action_entropy = action_dist.entropy().mean()

        return action_log_probs, action_entropy


class EntityActor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, max_entity_num, dropout=0.1):
        super().__init__()
        self.input_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.token_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.transformer_layer_1 = TransformerBlock(
            emb=hidden_dim,
            heads=1,
            mask=False,
            ff_hidden_mult=2,
            dropout=dropout
        )

        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, embs, entity_mask, agent_mask):
        original_embs = embs
        embs = self.input_net(embs)
        batch_size, n_entities, feat_size = embs.shape[0], embs.shape[1], embs.shape[2]

        h_mask = torch.ones(embs.shape[0], 1).to(embs.device)
        entity_mask = torch.cat([h_mask, entity_mask], dim=-1)
        entity_mask = entity_mask.unsqueeze(1).repeat(1, n_entities, 1, 1).reshape(
            batch_size * n_entities, n_entities + 1)

        embs = embs.unsqueeze(1).repeat(1, n_entities, 1, 1)
        h = self.token_layer(original_embs).unsqueeze(-2)
        embs = torch.cat([h, embs], dim=-2).reshape(batch_size * n_entities, n_entities + 1, -1)

        embs = self.transformer_layer_1(q=embs, k=embs, mask_q=entity_mask, mask_k=entity_mask)
        # embs = self.transformer_layer_2(q=embs, k=embs, mask_q=entity_mask, mask_k=entity_mask)
        embs = embs.reshape(batch_size, n_entities, -1, feat_size)
        embs = embs[:, :, 0, :]

        logits = self.output_net(embs)

        empty_action = torch.ones(1, logits.shape[-1]).to(logits.device) * -1e5
        empty_action[0, 0] = 1e5
        logits[agent_mask == 0] = empty_action

        action_dist = OneHotCategorical(logits=logits)
        action = action_dist.sample()

        return action, logits


class AttentionActor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, layers, activation=nn.ReLU):
        super().__init__()
        self.feedforward_model = build_model(hidden_size, out_dim, 1, hidden_size, activation)
        self._attention_stack = AttentionEncoder(1, hidden_size, hidden_size)
        self.embed = nn.Linear(in_dim, hidden_size)

    def forward(self, state_features):
        n_agents = state_features.shape[-2]
        batch_size = state_features.shape[:-2]
        embeds = F.relu(self.embed(state_features))
        embeds = embeds.view(-1, n_agents, embeds.shape[-1])
        attn_embeds = F.relu(self._attention_stack(embeds).view(*batch_size, n_agents, embeds.shape[-1]))
        x = self.feedforward_model(attn_embeds)
        action_dist = OneHotCategorical(logits=x)
        action = action_dist.sample()

        return action, x
