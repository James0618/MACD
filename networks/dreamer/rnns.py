import torch
import torch.nn as nn
from torch.distributions import OneHotCategorical

from configs.dreamer.DreamerAgentConfig import RSSMState
from networks.transformer.layers import AttentionEncoder


def stack_states(rssm_states: list, dim):
    return reduce_states(rssm_states, dim, torch.stack)


def cat_states(rssm_states: list, dim):
    return reduce_states(rssm_states, dim, torch.cat)


def reduce_states(rssm_states: list, dim, func):
    return RSSMState(*[func([getattr(state, key) for state in rssm_states], dim=dim)
                       for key in rssm_states[0].__dict__.keys()])


class DiscreteLatentDist(nn.Module):
    def __init__(self, in_dim, n_categoricals, n_classes, hidden_size, activation=nn.ReLU, norm=nn.Identity):
        super().__init__()
        self.n_categoricals = n_categoricals
        self.n_classes = n_classes
        self.dists = nn.Sequential(nn.Linear(in_dim, hidden_size),
                                   norm(hidden_size),
                                   activation(),
                                   nn.Linear(hidden_size, n_classes * n_categoricals))

    def forward(self, x, deterministic=False):
        logits = self.dists(x).view(x.shape[:-1] + (self.n_categoricals, self.n_classes))
        class_dist = OneHotCategorical(logits=logits)
        if deterministic:
            one_hot = torch.eye(self.n_classes)[class_dist.probs.argmax(dim=-1)].to(x.device)
        else:
            one_hot = class_dist.sample()
        latents = one_hot + class_dist.probs - class_dist.probs.detach()
        return logits.view(x.shape[:-1] + (-1,)), latents.view(x.shape[:-1] + (-1,))


class RSSMTransition(nn.Module):
    def __init__(self, config, hidden_size=200, activation=nn.ReLU, norm=nn.Identity):
        super().__init__()
        self._stoch_size = config.STOCHASTIC
        self._deter_size = config.DETERMINISTIC
        self._hidden_size = hidden_size
        self._activation = activation
        self._cell = nn.GRU(hidden_size, self._deter_size)
        self._norm = norm
        self._attention_stack = AttentionEncoder(3, hidden_size, hidden_size, dropout=0.1, norm=norm)
        self._rnn_input_model = self._build_rnn_input_model(config.ACTION_SIZE + self._stoch_size)
        self._stochastic_prior_model = DiscreteLatentDist(self._deter_size, config.N_CATEGORICALS, config.N_CLASSES,
                                                          self._hidden_size)

    def _build_rnn_input_model(self, in_dim):
        rnn_input_model = [nn.Linear(in_dim, self._hidden_size)]
        rnn_input_model += [self._norm(self._hidden_size)]
        rnn_input_model += [self._activation()]
        return nn.Sequential(*rnn_input_model)

    def forward(self, prev_actions, prev_states, mask=None):
        batch_size = prev_actions.shape[0]
        n_agents = prev_actions.shape[1]
        x = self._rnn_input_model(torch.cat([prev_actions, prev_states.stoch], dim=-1))
        x = self._attention_stack(x, mask=mask)
        x = self._cell(x.reshape(1, batch_size * n_agents, -1),
                       prev_states.deter.reshape(1, batch_size * n_agents, -1))[0].reshape(batch_size, n_agents, -1)
        logits, stoch_state = self._stochastic_prior_model(x)
        return RSSMState(logits=logits, stoch=stoch_state, deter=x)


class RSSMRepresentation(nn.Module):
    def __init__(self, config, transition_model: RSSMTransition):
        super().__init__()
        self._transition_model = transition_model
        self._stoch_size = config.STOCHASTIC
        self._deter_size = config.DETERMINISTIC
        self._stochastic_posterior_model = DiscreteLatentDist(
            self._deter_size + config.EMBED, config.N_CATEGORICALS, config.N_CLASSES, config.HIDDEN,
            activation=config.activation, norm=config.norm
        )

    def initial_state(self, batch_size, n_agents, **kwargs):
        return RSSMState(stoch=torch.zeros(batch_size, n_agents, self._stoch_size, **kwargs),
                         logits=torch.zeros(batch_size, n_agents, self._stoch_size, **kwargs),
                         deter=torch.zeros(batch_size, n_agents, self._deter_size, **kwargs))

    def forward(self, obs_embed, prev_actions, prev_states, mask=None, deterministic=False):
        """
        :param mask:
        :param deterministic:
        :param obs_embed: size(batch, n_agents, obs_size)
        :param prev_actions: size(batch, n_agents, action_size)
        :param prev_states: size(batch, n_agents, state_size)
        :return: RSSMState, global_state: size(batch, 1, global_state_size)
        """
        prior_states = self._transition_model(prev_actions, prev_states, mask)
        x = torch.cat([prior_states.deter, obs_embed], dim=-1)
        logits, stoch_state = self._stochastic_posterior_model(x, deterministic=deterministic)
        posterior_states = RSSMState(logits=logits, stoch=stoch_state, deter=prior_states.deter)
        return prior_states, posterior_states


def rollout_representation(representation_model, steps, obs_embed, action, prev_states, done, entity_mask):
    """
        Roll out the model with actions and observations from data.
        :param representation_model:
        :param done:
        :param entity_mask:
        :param steps: number of steps to roll out
        :param obs_embed: size(time_steps, batch_size, n_agents, embedding_size)
        :param action: size(time_steps, batch_size, n_agents, action_size)
        :param prev_states: RSSM state, size(batch_size, n_agents, state_size)
        :return: prior, posterior states. size(time_steps, batch_size, n_agents, state_size)
        """
    priors = []
    posteriors = []
    for t in range(steps):
        prior_states, posterior_states = representation_model(obs_embed[t], action[t], prev_states, entity_mask[t])
        prev_states = posterior_states.map(lambda x: x * (1.0 - done[t].unsqueeze(-1)))
        priors.append(prior_states)
        posteriors.append(posterior_states)

    prior = stack_states(priors, dim=0)
    post = stack_states(posteriors, dim=0)

    return prior.map(lambda x: x[:-1]), post.map(lambda x: x[:-1]), post.deter[1:]


def rollout_policy(transition_model, av_action, steps, policy, prev_state, entity_mask, agent_mask):
    """
        Roll out the model with a policy function.
        :param agent_mask:
        :param transition_model:
        :param av_action:
        :param entity_mask:
        :param steps: number of steps to roll out
        :param policy: RSSMState -> action
        :param prev_state: RSSM state, size(batch_size, state_size)
        :return: next states size(time_steps, batch_size, state_size),
                 actions size(time_steps, batch_size, action_size)
        """
    state = prev_state
    next_states = []
    actions = []
    av_actions = []
    entity_masks = []
    agent_masks = []
    policies = []
    for t in range(steps):
        feat = state.get_features().detach()
        action, pi = policy(feat, entity_mask.squeeze(-1), agent_mask.squeeze(-1))
        if av_action is not None:
            avail_actions = av_action(feat).sample()
            pi[avail_actions == 0] = -1e10
            action_dist = OneHotCategorical(logits=pi)
            action = action_dist.sample().squeeze(0)
            av_actions.append(avail_actions.squeeze(0))

        next_states.append(state)
        policies.append(pi)
        actions.append(action)
        entity_masks.append(entity_mask)
        agent_masks.append(agent_mask)
        state = transition_model(action, state, entity_mask)
    return {"imag_states": stack_states(next_states, dim=0),
            "actions": torch.stack(actions, dim=0),
            "av_actions": torch.stack(av_actions, dim=0) if len(av_actions) > 0 else None,
            "old_policy": torch.stack(policies, dim=0),
            "entity_mask": torch.stack(entity_masks, dim=0),
            "agent_mask": torch.stack(agent_masks, dim=0)}


def rollout_q_network(transition_model, av_action, steps, policy, prev_state, entity_mask, agent_mask):
    """
        Roll out the model with a policy function.
        :param agent_mask:
        :param transition_model:
        :param av_action:
        :param entity_mask:
        :param steps: number of steps to roll out
        :param policy: RSSMState -> action
        :param prev_state: RSSM state, size(batch_size, state_size)
        :return: next states size(time_steps, batch_size, state_size),
                 actions size(time_steps, batch_size, action_size)
        """
    state = prev_state
    next_states = []
    actions = []
    av_actions = []
    entity_masks = []
    agent_masks = []
    q_values = []
    for t in range(steps):
        feat = state.get_features().detach()
        q_value = policy(feat, entity_mask.squeeze(-1), agent_mask.squeeze(-1))
        if av_action is not None:
            avail_actions = av_action(feat).sample()
            q_value[avail_actions == 0] = -1e10

        ptr = torch.rand(q_value.shape[:-1]).unsqueeze(-1).to(q_value.device)
        action = torch.argmax(q_value, dim=-1, keepdim=True) * (ptr > 0.05).long() + \
            torch.randint(0, q_value.shape[-1], q_value.shape[:-1]).to(q_value.device).unsqueeze(-1) \
            * (ptr <= 0.05).long()

        next_states.append(state)
        q_values.append(q_value)
        actions.append(action)
        entity_masks.append(entity_mask)
        agent_masks.append(agent_mask)
        state = transition_model(action, state, entity_mask)

    return {"imag_states": stack_states(next_states, dim=0),
            "actions": torch.stack(actions, dim=0),
            "av_actions": torch.stack(av_actions, dim=0) if len(av_actions) > 0 else None,
            "q_values": torch.stack(q_values, dim=0),
            "entity_mask": torch.stack(entity_masks, dim=0),
            "agent_mask": torch.stack(agent_masks, dim=0)}
