import torch
import torch.nn as nn

from environments import Env
from networks.dreamer.dense import EntityRewardModel, EntityQModel, DecentralizedEntityBinaryModel
from networks.dreamer.vae import Encoder, Decoder
from networks.dreamer.transformers import RSSMRepresentation, RSSMTransition


class DreamerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.action_size = config.ACTION_SIZE

        self.entity_encoder = Encoder(in_dim=config.IN_DIM, hidden=config.HIDDEN, embed=config.EMBED)
        self.entity_decoder = Decoder(embed=config.FEAT, hidden=config.HIDDEN, out_dim=config.IN_DIM)

        self.transition = RSSMTransition(config, config.MODEL_HIDDEN)
        self.representation = RSSMRepresentation(config, self.transition)
        self.reward_model = EntityRewardModel(config=config)
        self.pcont = DecentralizedEntityBinaryModel(config.FEAT, config.REWARD_HIDDEN)

        if config.ENV_TYPE == Env.STARCRAFT:
            self.av_action = DecentralizedEntityBinaryModel(config.FEAT, config.PCONT_HIDDEN)
        else:
            self.av_action = None

        self.q_network = EntityQModel(config.HIDDEN, config.PCONT_HIDDEN, config.ACTION_SIZE)

    def forward(self, observations, prev_actions=None, prev_states=None, entity_mask=None):
        if prev_actions is None:
            prev_actions = torch.zeros(observations.size(0), observations.size(1), self.action_size,
                                       device=observations.device)
            prev_actions[:, :, 0] = 1.0

        if prev_states is None:
            prev_states = self.representation.initial_state(prev_actions.size(0), observations.size(1),
                                                            device=observations.device)

        return self.get_state_representation(observations, prev_actions, prev_states, entity_mask)

    def get_state_representation(self, observations, prev_actions, prev_states, entity_mask):
        """
        :param entity_mask: size(batch, n_entities)
        :param observations: size(batch, n_entities, in_dim)
        :param prev_actions: size(batch, n_entities, action_size)
        :param prev_states: size(batch, n_entities, state_size)
        :return: RSSMState
        """
        obs_embeds = self.entity_encoder(observations)
        _, states = self.representation(obs_embeds, prev_actions, prev_states, entity_mask)
        return states
