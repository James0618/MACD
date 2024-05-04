import torch
import torch.distributions as td
import torch.nn as nn

from networks.dreamer.utils import build_model, DiscDist
from networks.transformer.transformer_layers import TransformerBlock, MultiHeadAttention


class DenseModel(nn.Module):
    def __init__(self, in_dim, out_dim, layers, hidden, activation=nn.ELU, norm=nn.Identity):
        super().__init__()

        self.model = build_model(in_dim, out_dim, layers, hidden, activation, norm=norm, if_last_norm=True)

    def forward(self, features):
        return self.model(features)


class DenseBinaryModel(DenseModel):
    def __init__(self, in_dim, out_dim, layers, hidden, activation=nn.ELU, norm=nn.Identity):
        super().__init__(in_dim, out_dim, layers, hidden, activation, norm)

    def forward(self, features):
        dist_inputs = self.model(features)
        return td.independent.Independent(td.Bernoulli(logits=dist_inputs), 1)


class RewardModelV3(nn.Module):
    def __init__(self, config, activation=nn.ELU, norm=nn.Identity):
        super().__init__()
        self.config = config
        in_dim, out_dim = config.FEAT, config.DISCRETE_CLASSES
        hidden_dim, discrete_classes = config.REWARD_HIDDEN, config.DISCRETE_CLASSES
        self.model = build_model(in_dim, out_dim, config.REWARD_LAYERS, hidden_dim, activation, norm=norm)

    def forward(self, features):
        embs = self.model(features)
        dist = DiscDist(
            logits=embs, low=-10, high=10,
            device=self.config.DEVICE, discrete_number=self.config.DISCRETE_CLASSES
        )

        return dist


class EntityRewardModel(nn.Module):
    def __init__(self, config, ff_hidden_mult=4, dropout=0.1):
        super().__init__()
        self.config = config
        in_dim, hidden_dim, discrete_classes = config.FEAT, config.REWARD_HIDDEN, config.DISCRETE_CLASSES
        self.reward_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.input_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.transformer_layer_1 = TransformerBlock(
            emb=hidden_dim,
            heads=8,
            mask=False,
            ff_hidden_mult=ff_hidden_mult,
            dropout=dropout
        )
        self.transformer_layer_2 = TransformerBlock(
            emb=hidden_dim,
            heads=8,
            mask=False,
            ff_hidden_mult=ff_hidden_mult,
            dropout=dropout
        )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, discrete_classes)
        )

    def forward(self, embs, mask=None):
        embs = self.input_net(embs)
        h = self.reward_token.repeat(embs.shape[0], 1, 1).to(embs.device)
        h_mask = torch.ones(embs.shape[0], 1).to(embs.device)

        embs = torch.cat([h, embs], dim=-2)
        mask = torch.cat([h_mask, mask], dim=-1)
        embs = self.transformer_layer_1(q=embs, k=embs, mask_q=mask, mask_k=mask)
        embs = self.transformer_layer_2(q=embs, k=embs, mask_q=mask, mask_k=mask)
        embs = embs[:, 0, :]
        embs = self.output_net(embs)

        dist = DiscDist(
            logits=embs, low=-1.0, high=2.0,
            device=self.config.DEVICE, discrete_number=self.config.DISCRETE_CLASSES
        )

        return dist


class EntityQModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, ff_hidden_mult=4, dropout=0.0):
        super().__init__()
        self.input_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.transformer_layer_1 = TransformerBlock(
            emb=hidden_dim,
            heads=4,
            mask=False,
            ff_hidden_mult=ff_hidden_mult,
            dropout=dropout
        )
        # self.transformer_layer_2 = TransformerBlock(
        #     emb=in_dim,
        #     heads=4,
        #     mask=False,
        #     ff_hidden_mult=ff_hidden_mult,
        #     dropout=dropout
        # )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, embs, mask=None):
        embs = self.input_net(embs)
        embs = self.transformer_layer_1(q=embs, k=embs, mask_q=mask, mask_k=mask)
        # embs = self.transformer_layer_2(q=embs, k=embs, mask_q=mask, mask_k=mask)
        result = self.output_net(embs)

        return result


class DecentralizedEntityBinaryModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, ff_hidden_mult=4, dropout=0.0):
        super().__init__()
        self.reward_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.input_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.transformer_layer_1 = TransformerBlock(
            emb=hidden_dim,
            heads=4,
            mask=False,
            ff_hidden_mult=ff_hidden_mult,
            dropout=dropout
        )
        # self.transformer_layer_2 = TransformerBlock(
        #     emb=in_dim,
        #     heads=4,
        #     mask=False,
        #     ff_hidden_mult=ff_hidden_mult,
        #     dropout=dropout
        # )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, embs, mask=None):
        embs = self.input_net(embs)
        embs = self.transformer_layer_1(q=embs, k=embs, mask_q=mask, mask_k=mask)
        # embs = self.transformer_layer_2(q=embs, k=embs, mask_q=mask, mask_k=mask)

        result = self.output_net(embs)
        result = td.independent.Independent(td.Bernoulli(logits=result), 1)

        return result
