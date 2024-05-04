from copy import deepcopy
import numpy as np
import torch
import wandb
import torch.nn.functional as F
from torch.distributions import OneHotCategorical

from agent.optim.old_utils import rec_loss, compute_return, state_divergence_loss, calculate_ppo_loss, \
    batch_multi_agent, log_prob_loss, info_loss, compute_return_dae
from agent.utils.params import FreezeParameters
from networks.dreamer.old_rnns import rollout_representation, rollout_representation_with_mask, rollout_policy, \
    rollout_continuous_policy
from networks.dreamer.old_rnns import stack_states


def model_loss(config, model, obs, action, av_action, reward, done, fake, last, if_log=False, if_use_consistency=False):
    time_steps = obs.shape[0]
    batch_size = obs.shape[1]
    n_agents = obs.shape[2]

    embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
    embed = embed.reshape(time_steps, batch_size, n_agents, -1)

    prev_state = model.representation.initial_state(batch_size, n_agents, device=obs.device)
    prior, post, deters = rollout_representation(model.representation, time_steps, embed, action, prev_state, last)
    feat = torch.cat([post.stoch, deters], -1)
    feat_dec = post.get_features()

    if if_use_consistency:
        target = feat_dec.mean(-2, keepdim=True).detach()
        consistency_loss = 0.1 * (feat_dec - target).pow(2).sum(-1).mean()
    else:
        consistency_loss = 0.

    reconstruction_loss, i_feat = rec_loss(model.observation_decoder,
                                           feat_dec.reshape(-1, n_agents, feat_dec.shape[-1]),
                                           obs[:-1].reshape(-1, n_agents, obs.shape[-1]),
                                           1. - fake[:-1].reshape(-1, n_agents, 1))
    # reward_loss = F.smooth_l1_loss(model.reward_model(feat), reward[1:])
    reward_dist = model.reward_model(feat)
    reward_loss = - reward_dist.log_prob(reward[1:]).mean()
    reward_error = F.smooth_l1_loss(model.reward_model(feat).mode(), reward[1:])

    pcont_loss = log_prob_loss(model.pcont, feat, (1. - done[1:]))
    av_action_loss = log_prob_loss(model.av_action, feat_dec, av_action[:-1]) if av_action is not None else 0.
    i_feat = i_feat.reshape(time_steps - 1, batch_size, n_agents, -1)

    dis_loss = info_loss(i_feat[1:], model, action[1:-1], 1. - fake[1:-1].reshape(-1))
    div = state_divergence_loss(prior, post, config)

    total_loss = div + reward_loss + dis_loss + reconstruction_loss + pcont_loss + av_action_loss + consistency_loss
    if if_log:
        wandb.log({'Model/reward_loss': reward_error, 'Model/div': div, 'Model/av_action_loss': av_action_loss,
                   'Model/reconstruction_loss': reconstruction_loss, 'Model/info_loss': dis_loss,
                   'Model/pcont_loss': pcont_loss, 'Model/consistency_loss': consistency_loss})

    return total_loss


def central_model_loss(config, model, obs, central_obs, action, av_action, reward, done, fake, last, if_log=False,
                       if_use_consistency=False):
    time_steps = obs.shape[0]
    batch_size = obs.shape[1]
    n_agents = obs.shape[2]

    embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
    embed = embed.reshape(time_steps, batch_size, n_agents, -1)

    prev_state = model.representation.initial_state(batch_size, n_agents, device=obs.device)
    prior, post, deters = rollout_representation(model.representation, time_steps, embed, action, prev_state, last)
    feat = torch.cat([post.stoch, deters], -1)
    feat_dec = post.get_features()

    if if_use_consistency:
        target = feat_dec.mean(-2, keepdim=True).detach()
        consistency_loss = 0.1 * (feat_dec - target).pow(2).sum(-1).mean()
    else:
        consistency_loss = 0.

    reconstruction_loss, i_feat = rec_loss(model.observation_decoder,
                                           feat_dec.reshape(-1, n_agents, feat_dec.shape[-1]),
                                           central_obs[:-1].reshape(-1, n_agents, central_obs.shape[-1]),
                                           1. - fake[:-1].reshape(-1, n_agents, 1))
    # reward_loss = F.smooth_l1_loss(model.reward_model(feat), reward[1:])
    reward_dist = model.reward_model(feat)
    reward_loss = - reward_dist.log_prob(reward[1:]).mean()
    reward_error = F.smooth_l1_loss(model.reward_model(feat).mode(), reward[1:])

    pcont_loss = log_prob_loss(model.pcont, feat, (1. - done[1:]))
    av_action_loss = log_prob_loss(model.av_action, feat_dec, av_action[:-1]) if av_action is not None else 0.
    i_feat = i_feat.reshape(time_steps - 1, batch_size, n_agents, -1)

    dis_loss = info_loss(i_feat[1:], model, action[1:-1], 1. - fake[1:-1].reshape(-1))
    div = state_divergence_loss(prior, post, config)

    total_loss = div + reward_loss + dis_loss + reconstruction_loss + pcont_loss + av_action_loss + consistency_loss
    if if_log:
        wandb.log({'Model/reward_loss': reward_error, 'Model/div': div, 'Model/av_action_loss': av_action_loss,
                   'Model/reconstruction_loss': reconstruction_loss, 'Model/info_loss': dis_loss,
                   'Model/pcont_loss': pcont_loss, 'Model/consistency_loss': consistency_loss})

    return total_loss


def get_model_error(model, obs, action, reward, fake, last, mask):
    time_steps = obs.shape[0]
    batch_size = obs.shape[1]
    n_agents = obs.shape[2]

    with FreezeParameters([model]):
        embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
        embed = embed.reshape(time_steps, batch_size, n_agents, -1)

        prev_state = model.representation.initial_state(batch_size, n_agents, device=obs.device)
        prior, post, deters = rollout_representation_with_mask(
            model.representation, time_steps, embed, action, prev_state, last, mask
        )
        feat = torch.cat([post.stoch, deters], -1)
        feat_dec = post.get_features()
        x_pred, _ = model.observation_decoder(feat_dec.reshape(-1, n_agents, feat_dec.shape[-1]))
        recon_loss = F.smooth_l1_loss(x_pred, obs[:-1].reshape(-1, n_agents, obs.shape[-1]), reduction='none') * \
            (1. - fake[:-1].reshape(-1, n_agents, 1))
        recon_error = recon_loss.sum(-1).mean(-1).reshape(time_steps - 1, batch_size)
        reward_loss = F.smooth_l1_loss(model.reward_model(feat).mode(), reward[1:], reduction='none')
        reward_error = reward_loss.squeeze().mean(-1)

        # total_error = reward_error + recon_error
        total_error = reward_error

    return total_error, feat_dec


def continuous_actor_rollout(obs, action, last, model, actor, critic, config):
    n_agents = obs.shape[2]
    with FreezeParameters([model]):
        embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
        embed = embed.reshape(obs.shape[0], obs.shape[1], n_agents, -1)
        prev_state = model.representation.initial_state(obs.shape[1], obs.shape[2], device=obs.device)
        prior, post, _ = rollout_representation(model.representation, obs.shape[0], embed, action,
                                                prev_state, last)
        post = post.map(lambda x: x.reshape((obs.shape[0] - 1) * obs.shape[1], n_agents, -1))
        items = rollout_continuous_policy(model.transition, model.av_action, config.HORIZON, actor, post)
    imag_feat = items["imag_states"].get_features()
    imag_rew_feat = torch.cat([items["imag_states"].stoch[:-1], items["imag_states"].deter[1:]], -1)
    returns = critic_rollout(model, critic, imag_feat, imag_rew_feat, items["actions"],
                             items["imag_states"].map(lambda x: x.reshape(-1, n_agents, x.shape[-1])), config)

    output = [items["actions"][:-1].detach(),
              items["av_actions"][:-1].detach() if items["av_actions"] is not None else None,
              items["old_policy"][:-1].detach(), imag_feat[:-1].detach(), returns.detach()]
    output_states = items["imag_states"].map(lambda x: x[:-1].reshape(-1, n_agents, x.shape[-1]))

    return [batch_multi_agent(v, n_agents) for v in output] + [output_states]


def actor_rollout(obs, action, last, model, actor, critic, config):
    n_agents = obs.shape[2]
    with FreezeParameters([model]):
        embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
        embed = embed.reshape(obs.shape[0], obs.shape[1], n_agents, -1)
        prev_state = model.representation.initial_state(obs.shape[1], obs.shape[2], device=obs.device)
        prior, post, _ = rollout_representation(model.representation, obs.shape[0], embed, action,
                                                prev_state, last)
        post = post.map(lambda x: x.reshape((obs.shape[0] - 1) * obs.shape[1], n_agents, -1))
        items = rollout_policy(model.transition, model.av_action, config.HORIZON, actor, post)
    imag_feat = items["imag_states"].get_features()
    imag_rew_feat = torch.cat([items["imag_states"].stoch[:-1], items["imag_states"].deter[1:]], -1)
    returns = critic_rollout(model, critic, imag_feat, imag_rew_feat, items["actions"],
                             items["imag_states"].map(lambda x: x.reshape(-1, n_agents, x.shape[-1])), config)

    output = [items["actions"][:-1].detach(),
              items["av_actions"][:-1].detach() if items["av_actions"] is not None else None,
              items["old_policy"][:-1].detach(), imag_feat[:-1].detach(), returns.detach()]
    output_states = items["imag_states"].map(lambda x: x[:-1].reshape(-1, n_agents, x.shape[-1]))

    return [batch_multi_agent(v, n_agents) for v in output] + [output_states]


def shapley_actor_rollout(start_states, group, origin_actions, model, actor, critic, config):
    n_agents = origin_actions.shape[-2]
    noop_action = torch.zeros_like(origin_actions)
    assert model.av_action is None

    with FreezeParameters([model, actor, critic]):
        state = start_states
        next_states = []
        actions = []
        imag_rewards = []
        for t in range(config.HORIZON):
            feat = state.get_features().detach()
            if t == 0:
                action = origin_actions
                action = action * group + noop_action * (1 - group)
            else:
                action, _, _, _ = actor(feat)

            next_states.append(state)
            next_state = model.transition(action, state)
            imag_rew_feat = torch.cat([state.stoch, next_state.deter], -1)
            imag_reward = calculate_reward(model, imag_rew_feat)
            imag_rewards.append(imag_reward)
            state = next_state

        next_states.append(state)
        items = {
            "imag_states": stack_states(next_states, dim=0),
            "imag_rewards": torch.stack(imag_rewards, dim=0),
        }

    imag_feat = items["imag_states"].get_features()
    imag_rew_feat = torch.cat([items["imag_states"].stoch[:-1], items["imag_states"].deter[1:]], -1)
    with FreezeParameters([model, critic]):
        imag_rewards = items['imag_rewards'].reshape(items["imag_rewards"].shape[:-1]).unsqueeze(-1).mean(
            -2, keepdim=True)
        value_dist = critic(imag_feat, actions)
        value = value_dist.mode()
        discount_arr = model.pcont(imag_rew_feat).mean

    next_values = torch.cat([value[1:-1], value[-1][None]], 0)
    target = imag_rewards + config.GAMMA * discount_arr * next_values * (1 - config.DISCOUNT_LAMBDA)
    outputs = []
    accumulated_reward = value[-1]
    for t in reversed(range(imag_rewards.shape[0])):
        discount_factor = discount_arr[t]
        accumulated_reward = target[t] + config.GAMMA * discount_factor * accumulated_reward * config.DISCOUNT_LAMBDA
        outputs.append(accumulated_reward)
    returns = torch.flip(torch.stack(outputs), [0])[0]

    return batch_multi_agent(returns.detach(), n_agents)


def dae_actor_rollout(obs, action, last, model, actor, critic, config):
    n_agents = obs.shape[2]
    with FreezeParameters([model]):
        embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
        embed = embed.reshape(obs.shape[0], obs.shape[1], n_agents, -1)
        prev_state = model.representation.initial_state(obs.shape[1], obs.shape[2], device=obs.device)
        prior, post, _ = rollout_representation(model.representation, obs.shape[0], embed, action,
                                                prev_state, last)
        post = post.map(lambda x: x.reshape((obs.shape[0] - 1) * obs.shape[1], n_agents, -1))
        items = rollout_policy(model.transition, model.av_action, config.HORIZON, actor, post)
    imag_feat = items["imag_states"].get_features()
    imag_rew_feat = torch.cat([items["imag_states"].stoch[:-1], items["imag_states"].deter[1:]], -1)
    advs, returns = dae_critic_rollout(
        model, actor, critic, imag_feat, imag_rew_feat, items["actions"],
        items["imag_states"].map(lambda x: x.reshape(-1, n_agents, x.shape[-1])), config
    )

    output = [items["actions"][:-1].detach(),
              items["av_actions"][:-1].detach() if items["av_actions"] is not None else None,
              items["old_policy"][:-1].detach(), imag_feat[:-1].detach(), returns.detach(), advs.detach()]
    output_states = items["imag_states"].map(lambda x: x[:-1].reshape(-1, n_agents, x.shape[-1]))

    return [batch_multi_agent(v, n_agents) for v in output] + [output_states]


def critic_rollout(model, critic, states, rew_states, actions, raw_states, config, if_shapley=False):
    with FreezeParameters([model, critic]):
        imag_reward = calculate_next_reward(model, actions, raw_states)
        imag_reward = imag_reward.reshape(actions.shape[:-1]).unsqueeze(-1).mean(-2, keepdim=True)[:-1]
        # rewards, imag_reward = calculate_next_reward_dr(model, actions, raw_states)
        # imag_reward = imag_reward.reshape(actions.shape[:-1]).unsqueeze(-1)[:-1]
        value_dist = critic(states, actions)
        value = value_dist.mode()
        discount_arr = model.pcont(rew_states).mean
        if not if_shapley:
            wandb.log({'Value/Max reward': imag_reward.max(), 'Value/Min reward': imag_reward.min(),
                       'Value/Reward': imag_reward.mean(), 'Value/Discount': discount_arr.mean(),
                       'Value/Value': value.mean()})
    returns = compute_return(imag_reward, value[:-1], discount_arr, bootstrap=value[-1], lmbda=config.DISCOUNT_LAMBDA,
                             gamma=config.GAMMA)
    return returns


def dae_critic_rollout(model, actor, critic, states, rew_states, actions, raw_states, config, if_shapley=False):
    with FreezeParameters([model, actor, critic]):
        imag_reward = calculate_next_reward(model, actions, raw_states)
        imag_reward = imag_reward.reshape(actions.shape[:-1]).unsqueeze(-1).mean(-2, keepdim=True)[:-1]
        value_dist = critic(states, actions)
        value = value_dist.mode()
        discount_arr = model.pcont(rew_states).mean
        if not if_shapley:
            wandb.log({'Value/Max reward': imag_reward.max(), 'Value/Min reward': imag_reward.min(),
                       'Value/Reward': imag_reward.mean(), 'Value/Discount': discount_arr.mean(),
                       'Value/Value': value.mean()})

        agent_num, act_size = actions.shape[-2], actions.shape[-1]
        dae_rewards = torch.zeros(imag_reward.shape[0], imag_reward.shape[1], agent_num, 1).to(imag_reward.device)
        temp_states = deepcopy(raw_states)
        _, new_policy = actor(temp_states.get_features())
        new_policy = new_policy.reshape(actions.shape)
        for i in range(agent_num):
            temp_rewards = []
            for j in range(act_size):
                current_action = torch.zeros(act_size).to(actions.device)
                current_action[j] = 1
                current_action = current_action.unsqueeze(0).unsqueeze(0).repeat(actions.shape[0], actions.shape[1], 1)
                temp_actions = deepcopy(actions)
                temp_actions[:, :, i] = current_action
                temp_reward = calculate_next_reward(model, temp_actions, temp_states)
                temp_reward = temp_reward.reshape(actions.shape[:-1]).mean(-1, keepdim=True)[:-1]
                temp_rewards.append(temp_reward)

            pi = torch.softmax(new_policy[:, :, i], -1)[:-1]
            dae_rewards[:, :, i] = (torch.stack(temp_rewards, dim=-1)[:, :, 0] * pi).sum(-1, keepdim=True)

    advs, returns = compute_return_dae(
        imag_reward, dae_rewards, value[:-1], discount_arr, bootstrap=value[-1], lmbda=config.DISCOUNT_LAMBDA,
        gamma=config.GAMMA
    )

    return advs, returns


# def calculate_reward(model, states, mask=None):
#     imag_reward = model.reward_model(states)
#     if mask is not None:
#         imag_reward *= mask
#     return imag_reward


def calculate_reward(model, states, mask=None):
    batch_size, n_agents, feat_size = states.shape
    states = states.reshape(-1, feat_size)
    reward_dist = model.reward_model(states)
    imag_reward = reward_dist.mode()
    if mask is not None:
        imag_reward *= mask

    imag_reward = imag_reward.reshape(batch_size, n_agents, 1)

    return imag_reward


def calculate_next_reward(model, actions, states):
    actions = actions.reshape(-1, actions.shape[-2], actions.shape[-1])
    next_state = model.transition(actions, states)
    imag_rew_feat = torch.cat([states.stoch, next_state.deter], -1)
    return calculate_reward(model, imag_rew_feat)


def calculate_next_reward_dr(model, actions, states):
    # with difference rewards
    actions = actions.reshape(-1, actions.shape[-2], actions.shape[-1])
    next_state = model.transition(actions, states)
    imag_rew_feat = torch.cat([states.stoch, next_state.deter], -1)
    rewards = calculate_reward(model, imag_rew_feat).mean(-2, keepdim=True).repeat(1, actions.shape[-2], 1)

    dr_states = states.map(lambda x: x.unsqueeze(1).repeat(1, actions.shape[-2], 1, 1))
    dr_actions = actions.unsqueeze(1).repeat(1, actions.shape[-2], 1, 1)
    default_action = torch.zeros(actions.shape[-1]).to(actions.device)
    default_action[0] = 1
    for i in range(actions.shape[-2]):
        dr_actions[:, i, i] = default_action

    dr_actions = dr_actions.reshape(-1, actions.shape[-2], actions.shape[-1])
    dr_states = dr_states.map(lambda x: x.reshape(-1, x.shape[-2], x.shape[-1]))

    dr_next_state = model.transition(dr_actions, dr_states)
    dr_rew_feat = torch.cat([dr_states.stoch, dr_next_state.deter], -1)
    difference_rewards = calculate_reward(model, dr_rew_feat)
    difference_rewards = difference_rewards.reshape(-1, actions.shape[-2], actions.shape[-2], 1)
    difference_rewards = difference_rewards.squeeze(-1).mean(-1, keepdim=True)

    difference = rewards - difference_rewards
    difference = (difference - difference.min()) / (difference.max() - difference.min() + 1e-10)
    difference = torch.softmax(difference * 3, dim=-2) * rewards

    return rewards, difference


def actor_loss(imag_states, actions, av_actions, old_policy, advantage, actor, ent_weight):
    _, new_policy = actor(imag_states)
    if av_actions is not None:
        new_policy[av_actions == 0] = -1e10
    actions = actions.argmax(-1, keepdim=True)
    rho = (F.log_softmax(new_policy, dim=-1).gather(2, actions) -
           F.log_softmax(old_policy, dim=-1).gather(2, actions)).exp()
    ppo_loss, ent_loss = calculate_ppo_loss(new_policy, rho, advantage)
    if np.random.randint(10) == 9:
        wandb.log({'Policy/Entropy': ent_loss.mean(), 'Policy/Mean action': actions.float().mean()})

    return (ppo_loss + ent_loss.unsqueeze(-1) * ent_weight).mean()


def ac_actor_loss(imag_states, actions, av_actions, advantage, actor, ent_weight):
    assert av_actions is None
    new_log_probs, dist_entropy = actor.evaluate_actions(imag_states, actions)
    actions = actions.argmax(-1, keepdim=True)
    ac_loss = advantage
    if np.random.randint(10) == 9:
        wandb.log({'Policy/Entropy': dist_entropy.mean(), 'Policy/Mean action': actions.float().mean()})
    loss = - (ac_loss + dist_entropy.unsqueeze(-1) * ent_weight).mean()

    return loss


def continuous_actor_loss(imag_states, actions, av_actions, old_policy, advantage, actor, ent_weight, eps=0.2):
    assert av_actions is None
    new_log_probs, dist_entropy = actor.evaluate_actions(imag_states, actions)
    rho = torch.exp(new_log_probs - old_policy)
    ppo_loss = - torch.min(rho * advantage, rho.clamp(1 - eps, 1 + eps) * advantage)
    ent_loss = - dist_entropy
    if np.random.randint(10) == 9:
        wandb.log({'Policy/Entropy': dist_entropy.mean(), 'Policy/Mean action': actions.float().mean()})

    return (ppo_loss + ent_loss.unsqueeze(-1) * ent_weight).mean()


def value_loss(critic, old_critic, actions, imag_feat, targets):
    value_dist = critic(imag_feat, actions)
    loss = - value_dist.log_prob(targets)

    # old_target = old_critic(imag_feat.detach(), actions).mode()
    # loss -= value_dist.log_prob(old_target.detach())

    value_error = (value_dist.mode() - targets.detach()).pow(2)

    return torch.mean(loss), torch.mean(value_error).detach()


def ac_value_loss(critic, old_critic, actions, imag_feat, targets):
    value_dist = critic(imag_feat, actions)
    loss = - value_dist.log_prob(targets)

    old_target = old_critic(imag_feat.detach(), actions).mode()
    loss -= value_dist.log_prob(old_target.detach())

    value_error = (value_dist.mode() - targets.detach()).pow(2)

    return torch.mean(loss), torch.mean(value_error).detach()
