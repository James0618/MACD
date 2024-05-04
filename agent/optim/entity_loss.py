import numpy as np
import torch
import wandb
import torch.nn.functional as F

from agent.optim.utils import rec_loss, compute_return, state_divergence_loss, calculate_ppo_loss, \
    batch_multi_agent, entity_log_prob_loss, entity_info_loss, rew_loss
from agent.utils.params import FreezeParameters
from networks.dreamer.rnns import rollout_representation, rollout_policy, rollout_q_network


def model_loss(config, model, obs, action, av_action, reward, done, agent_mask, entity_mask, last):
    time_steps = obs.shape[0]
    batch_size = obs.shape[1]
    n_entities = obs.shape[2]

    embed = model.entity_encoder(obs.reshape(-1, n_entities, obs.shape[-1]))
    embed = embed.reshape(time_steps, batch_size, n_entities, -1)

    prev_state = model.representation.initial_state(batch_size, n_entities, device=obs.device)
    prior, post, deters = rollout_representation(
        model.representation, time_steps, embed, action, prev_state, last, entity_mask)
    feat = torch.cat([post.stoch, deters], -1)      # a trick which is not mentioned in the paper
    feat_dec = post.get_features()

    reconstruction_loss, i_feat = rec_loss(model.entity_decoder,
                                           feat_dec.reshape(-1, n_entities, feat_dec.shape[-1]),
                                           obs[:-1].reshape(-1, n_entities, obs.shape[-1]),
                                           entity_mask[:-1].reshape(-1, n_entities, 1))

    reward_loss = rew_loss(
        model=model, feat=feat.reshape(feat.shape[0] * feat.shape[1], feat.shape[2], -1),
        reward=reward[1:].reshape(-1, 1), entity_mask=entity_mask[:-1].reshape(feat.shape[0] * feat.shape[1], -1),
        config=config
    )
    # reward_loss = F.smooth_l1_loss(
    #     model.reward_model(feat.detach().reshape(feat.shape[0] * feat.shape[1], feat.shape[2], -1),
    #                        mask=entity_mask[:-1].reshape(feat.shape[0] * feat.shape[1], -1)),
    #     reward[1:].reshape(feat.shape[0] * feat.shape[1], -1)
    # )
    pcont_loss = entity_log_prob_loss(
        model.pcont, feat.reshape(feat.shape[0] * feat.shape[1], feat.shape[2], -1),
        mask=entity_mask[:-1].reshape(feat.shape[0] * feat.shape[1], -1),
        target=entity_mask[1:].reshape(feat.shape[0] * feat.shape[1], feat.shape[2], -1)
    )
    av_action_loss = 0.

    i_feat = i_feat.reshape(time_steps - 1, batch_size, n_entities, -1)
    info_loss = entity_info_loss(
        i_feat[1:].reshape((i_feat.shape[0] - 1) * i_feat.shape[1], i_feat.shape[2], -1), model,
        action[1:-1].reshape((i_feat.shape[0] - 1) * i_feat.shape[1], i_feat.shape[2], -1),
        entity_mask[1:-1].reshape((i_feat.shape[0] - 1) * i_feat.shape[1], -1),
        agent_mask[1:-1].reshape((i_feat.shape[0] - 1) * i_feat.shape[1], -1),
    )
    div = state_divergence_loss(prior, post, config)

    total_loss = div + reward_loss + info_loss + reconstruction_loss + pcont_loss + av_action_loss

    loss_info = {
        'reward_loss': reward_loss.detach().cpu().float(),
        'reconstruction_loss': reconstruction_loss.detach().cpu().float(),
        'info_loss': info_loss.detach().cpu().float(), 'pcont_loss': pcont_loss.detach().cpu().float(),
        'div': div.detach().cpu().float()
    }

    return total_loss, loss_info


def q_network_rollout(obs, action, last, entity_mask, agent_mask, model, q_network, target_q_network, config):
    n_agents = obs.shape[2]
    with FreezeParameters([model]):
        embed = model.entity_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
        embed = embed.reshape(obs.shape[0], obs.shape[1], n_agents, -1)
        prev_state = model.representation.initial_state(obs.shape[1], obs.shape[2], device=obs.device)
        prior, post, _ = rollout_representation(
            model.representation, obs.shape[0], embed, action, prev_state, last, entity_mask)
        post = post.map(lambda x: x.reshape((obs.shape[0] - 1) * obs.shape[1], n_agents, -1))
        items = rollout_q_network(
            model.transition, model.av_action, config.HORIZON, q_network, post,
            entity_mask[:-1].reshape((obs.shape[0] - 1) * obs.shape[1], n_agents, -1),
            agent_mask[:-1].reshape((obs.shape[0] - 1) * obs.shape[1], n_agents, -1)
        )

    imag_feat = items["imag_states"].get_features()
    imag_rew_feat = torch.cat([items["imag_states"].stoch[:-1], items["imag_states"].deter[1:]], -1)

    output = [items["actions"][:-1].detach(),
              items["av_actions"][:-1].detach() if items["av_actions"] is not None else None,
              items["old_policy"][:-1].detach(), imag_feat[:-1].detach(),
              items['entity_mask'][:-1].detach(),
              items['agent_mask'][:-1].detach()]

    return [batch_multi_agent(v, n_agents) for v in output]


def actor_rollout(obs, action, last, entity_mask, agent_mask, model, actor, critic, config):
    n_agents = obs.shape[2]
    with FreezeParameters([model]):
        embed = model.entity_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
        embed = embed.reshape(obs.shape[0], obs.shape[1], n_agents, -1)
        prev_state = model.representation.initial_state(obs.shape[1], obs.shape[2], device=obs.device)
        prior, post, _ = rollout_representation(
            model.representation, obs.shape[0], embed, action, prev_state, last, entity_mask)
        post = post.map(lambda x: x.reshape((obs.shape[0] - 1) * obs.shape[1], n_agents, -1))
        items = rollout_policy(model.transition, model.av_action, config.HORIZON, actor, post,
                               entity_mask[:-1].reshape((obs.shape[0] - 1) * obs.shape[1], n_agents, -1),
                               agent_mask[:-1].reshape((obs.shape[0] - 1) * obs.shape[1], n_agents, -1))
    imag_feat = items["imag_states"].get_features()
    imag_rew_feat = torch.cat([items["imag_states"].stoch[:-1], items["imag_states"].deter[1:]], -1)

    returns = critic_rollout(model, critic, imag_feat, imag_rew_feat, items["actions"],
                             items["imag_states"].map(lambda x: x.reshape(-1, n_agents, x.shape[-1])),
                             items["entity_mask"], items["agent_mask"], config).unsqueeze(-1)
    output = [items["actions"][:-1].detach(),
              items["av_actions"][:-1].detach() if items["av_actions"] is not None else None,
              items["old_policy"][:-1].detach(), imag_feat[:-1].detach(),
              returns.detach(),
              items['entity_mask'][:-1].detach(),
              items['agent_mask'][:-1].detach()]
    return [batch_multi_agent(v, n_agents) for v in output]


def critic_rollout(model, critic, states, rew_states, actions, raw_states, entity_mask, agent_mask, config):
    n_entities = entity_mask.shape[2]
    with FreezeParameters([model, critic]):
        imag_reward = calculate_next_reward(model, actions, raw_states,
                                            entity_mask.reshape(entity_mask.shape[0] * entity_mask.shape[1], -1))
        imag_reward = imag_reward.reshape(actions.shape[:-2]).unsqueeze(-1)[:-1]
        value = critic(states, entity_mask, agent_mask)
        discount_arr = model.pcont(
            rew_states.reshape((entity_mask.shape[0] - 1) * entity_mask.shape[1], n_entities, -1),
            entity_mask[:-1].reshape((entity_mask.shape[0] - 1) * entity_mask.shape[1], -1)
        ).mean.reshape((entity_mask.shape[0] - 1), entity_mask.shape[1], n_entities)

        wandb.log({'Value/Max reward': imag_reward.max(), 'Value/Min reward': imag_reward.min(),
                   'Value/Reward': imag_reward.mean(), 'Value/Discount': discount_arr.mean(),
                   'Value/Value': value.mean()})

    returns = compute_return(imag_reward, value[:-1].squeeze(-1), discount_arr, bootstrap=value[-1].squeeze(-1),
                             lmbda=config.DISCOUNT_LAMBDA, gamma=config.GAMMA)
    return returns


def calculate_reward(model, states, entity_mask):
    config = model.config
    reward_dist = model.reward_model(states, entity_mask)
    # imag_reward = imag_reward.argmax(-1, keepdim=True) / (config.DISCRETE_CLASSES - 1)
    # imag_reward = imag_reward * (config.REWARD_MAX - config.REWARD_MIN) + config.REWARD_MIN
    imag_reward = reward_dist.mode()

    return imag_reward


def calculate_next_reward(model, actions, states, entity_mask):
    actions = actions.reshape(-1, actions.shape[-2], actions.shape[-1])
    next_state = model.transition(actions, states)
    imag_rew_feat = torch.cat([states.stoch, next_state.deter], -1)

    return calculate_reward(model, imag_rew_feat, entity_mask)


def actor_loss(imag_states, actions, av_actions, old_policy, advantage, actor, ent_weight, entity_mask, agent_mask):
    _, new_policy = actor(imag_states, entity_mask.squeeze(-1), agent_mask.squeeze(-1))
    new_policy = new_policy
    if av_actions is not None:
        new_policy[av_actions == 0] = -1e10
    actions = actions.argmax(-1, keepdim=True)
    rho = (F.log_softmax(new_policy, dim=-1).gather(2, actions) -
           F.log_softmax(old_policy, dim=-1).gather(2, actions)).exp()
    ppo_loss, ent_loss = calculate_ppo_loss(new_policy, rho, advantage, agent_mask)
    if np.random.randint(10) == 9:
        wandb.log({'Policy/Entropy': ent_loss.mean(), 'Policy/Mean action': actions.float().mean()})
    return (ppo_loss + ent_loss.unsqueeze(-1) * ent_weight).mean()


def value_loss(critic, actions, imag_feat, targets, entity_mask, agent_mask):
    value_pred = critic(imag_feat, entity_mask, agent_mask)
    mse_loss = (targets * agent_mask - value_pred) ** 2 / 2.0
    return torch.mean(mse_loss)
