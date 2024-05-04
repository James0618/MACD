import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.dreamer.utils import build_model, DiscDist
from networks.transformer.layers import AttentionEncoder
from networks.transformer.transformer_layers import TransformerBlock


class Critic(nn.Module):
    def __init__(self, in_dim, hidden_size, layers=2, activation=nn.ELU):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.activation = activation
        self.feedforward_model = build_model(in_dim, 1, layers, hidden_size, activation)

    def forward(self, state_features, actions):
        return self.feedforward_model(state_features)


class MADDPGCritic(nn.Module):
    def __init__(self, in_dim, hidden_size, activation=nn.ReLU):
        super().__init__()
        self.feedforward_model = build_model(hidden_size, 1, 1, hidden_size, activation)
        self._attention_stack = AttentionEncoder(1, hidden_size, hidden_size)
        self.embed = nn.Linear(in_dim, hidden_size)
        self.prior = build_model(in_dim, 1, 3, hidden_size, activation)

    def forward(self, state_features, actions):
        n_agents = state_features.shape[-2]
        batch_size = state_features.shape[:-2]
        embeds = F.relu(self.embed(state_features))
        embeds = embeds.view(-1, n_agents, embeds.shape[-1])
        attn_embeds = F.relu(self._attention_stack(embeds).view(*batch_size, n_agents, embeds.shape[-1]))
        return self.feedforward_model(attn_embeds)


class DreamerV3Critic(nn.Module):
    def __init__(self, in_dim, hidden_size, config=None, activation=nn.ReLU, norm=nn.Identity):
        super().__init__()
        if activation == nn.ReLU:
            self.activation = 'relu'
        elif activation == nn.SiLU:
            self.activation = 'silu'
        else:
            raise NotImplementedError

        self._attention_stack = AttentionEncoder(1, hidden_size, hidden_size, norm=norm)
        self.discrete_classes = config.DISCRETE_CLASSES
        self.device = config.DEVICE
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            norm(hidden_size)
        )
        self.prior = build_model(in_dim, 1, 3, hidden_size, activation, norm=norm)
        self.feedforward_model = build_model(
            hidden_size, self.discrete_classes, 1, hidden_size, activation, norm=norm, if_last_norm=True
        )

    def forward(self, state_features, actions):
        n_agents = state_features.shape[-2]
        batch_size = state_features.shape[:-2]
        if self.activation == 'relu':
            embeds = F.relu(self.embed(state_features))
        elif self.activation == 'silu':
            embeds = F.silu(self.embed(state_features))
        else:
            raise NotImplementedError
        embeds = embeds.view(-1, n_agents, embeds.shape[-1])
        if self.activation == 'relu':
            attn_embeds = F.relu(self._attention_stack(embeds).view(*batch_size, n_agents, embeds.shape[-1]))
        elif self.activation == 'silu':
            attn_embeds = F.silu(self._attention_stack(embeds).view(*batch_size, n_agents, embeds.shape[-1]))
        else:
            raise NotImplementedError

        value_embs = self.feedforward_model(attn_embeds)

        value_dist = DiscDist(
            logits=value_embs, low=-10, high=10,
            device=self.device, discrete_number=self.discrete_classes
        )

        return value_dist


class ShapleyCritic(nn.Module):
    def __init__(self, in_dim, hidden_size, config=None, activation=nn.ReLU, norm=nn.Identity):
        super().__init__()
        if activation == nn.ReLU:
            self.activation = 'relu'
        elif activation == nn.SiLU:
            self.activation = 'silu'
        else:
            raise NotImplementedError

        self.value_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self._attention_stack = AttentionEncoder(1, hidden_size, hidden_size, norm=norm)
        self.discrete_classes = config.DISCRETE_CLASSES
        self.device = config.DEVICE
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            norm(hidden_size)
        )
        self.feedforward_model = build_model(
            hidden_size, self.discrete_classes, 1, hidden_size, activation, norm=norm, if_last_norm=True
        )

    def forward(self, state_features, actions, central=False):
        n_agents = state_features.shape[-2]
        batch_size = state_features.shape[:-2]
        if self.activation == 'relu':
            embeds = F.relu(self.embed(state_features))
        elif self.activation == 'silu':
            embeds = F.silu(self.embed(state_features))
        else:
            raise NotImplementedError
        embeds = embeds.view(-1, n_agents, embeds.shape[-1])
        value_token = self.value_token.repeat(embeds.shape[0], 1, 1)
        embeds = torch.cat([value_token, embeds], dim=1)
        if self.activation == 'relu':
            attn_embeds = F.relu(self._attention_stack(embeds))
        elif self.activation == 'silu':
            attn_embeds = F.silu(self._attention_stack(embeds))
        else:
            raise NotImplementedError

        if central:
            value_embs = self.feedforward_model(attn_embeds)[:, :1, :]
            value_embs = value_embs.view(*batch_size, 1, value_embs.shape[-1])
            value_dist = DiscDist(
                logits=value_embs, low=-10, high=10,
                device=self.device, discrete_number=self.discrete_classes
            )
        else:
            value_embs = self.feedforward_model(attn_embeds)[:, :1, :].repeat(1, n_agents, 1)
            value_embs = value_embs.view(*batch_size, n_agents, value_embs.shape[-1])
            value_dist = DiscDist(
                logits=value_embs, low=-10, high=10,
                device=self.device, discrete_number=self.discrete_classes
            )

        return value_dist


class EntityPPOCritic(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.value_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.input_net = nn.Sequential(
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
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, embs, entity_mask, agent_mask):
        if len(embs.shape) == 4:
            embs = self.input_net(embs)
            seq_length, batch_size, n_entities, feat_size = embs.shape[0], embs.shape[1], embs.shape[2], embs.shape[3]
            embs = embs.reshape(seq_length * batch_size, n_entities, feat_size)
            entity_mask = entity_mask.reshape(seq_length * batch_size, n_entities)
            agent_mask = agent_mask.reshape(seq_length * batch_size, n_entities)

            h = self.value_token.repeat(seq_length * batch_size, 1, 1).to(embs.device)
            h_mask = torch.ones(seq_length * batch_size, 1).to(embs.device)
            embs = torch.cat([h, embs], dim=-2)
            entity_mask = torch.cat([h_mask, entity_mask], dim=-1)

            embs = embs.unsqueeze(1).repeat(1, n_entities, 1, 1).reshape(
                seq_length * batch_size * n_entities, n_entities + 1, -1)
            entity_mask = entity_mask.unsqueeze(1).repeat(1, n_entities, 1, 1).reshape(
                seq_length * batch_size * n_entities, n_entities + 1)

            embs = self.transformer_layer_1(q=embs, k=embs, mask_q=entity_mask, mask_k=entity_mask)
            embs = embs.reshape(seq_length * batch_size, n_entities, -1, feat_size)
            embs = embs[:, :, 0, :]

            value = self.output_net(embs)
            value = value * agent_mask.unsqueeze(-1)

            value = value.reshape(seq_length, batch_size, n_entities, 1)

        elif len(embs.shape) == 3:
            embs = self.input_net(embs)
            batch_size, n_entities, feat_size = embs.shape[0], embs.shape[1], embs.shape[2]
            embs = embs.reshape(batch_size, n_entities, feat_size)
            entity_mask = entity_mask.reshape(batch_size, n_entities)
            agent_mask = agent_mask.reshape(batch_size, n_entities)

            h = self.value_token.repeat(batch_size, 1, 1).to(embs.device)
            h_mask = torch.ones(batch_size, 1).to(embs.device)
            embs = torch.cat([h, embs], dim=-2)
            entity_mask = torch.cat([h_mask, entity_mask], dim=-1)

            embs = embs.unsqueeze(1).repeat(1, n_entities, 1, 1).reshape(
                batch_size * n_entities, n_entities + 1, -1)
            entity_mask = entity_mask.unsqueeze(1).repeat(1, n_entities, 1, 1).reshape(
                batch_size * n_entities, n_entities + 1)

            embs = self.transformer_layer_1(q=embs, k=embs, mask_q=entity_mask, mask_k=entity_mask)
            # embs = self.transformer_layer_2(q=embs, k=embs, mask_q=mask, mask_k=mask)
            embs = embs.reshape(batch_size, n_entities, -1, feat_size)
            embs = embs[:, :, 0, :]

            value = self.output_net(embs)
            value = value * agent_mask.unsqueeze(-1)

        else:
            raise NotImplementedError

        return value
