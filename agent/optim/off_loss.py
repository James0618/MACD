import numpy as np
import torch
import wandb
import torch.nn.functional as F

from agent.optim.old_utils import rec_loss, compute_return, state_divergence_loss, calculate_ppo_loss, \
    batch_multi_agent, log_prob_loss, info_loss
from agent.utils.params import FreezeParameters
from networks.dreamer.sac_rnns import rollout_representation, rollout_policy, rollout_q


def model_loss(config, model, obs, action, av_action, reward, done, fake, last, if_log=False):
    time_steps = obs.shape[0]
    batch_size = obs.shape[1]
    n_agents = obs.shape[2]

    embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
    embed = embed.reshape(time_steps, batch_size, n_agents, -1)

    prev_state = model.representation.initial_state(batch_size, n_agents, device=obs.device)
    prior, post, deters = rollout_representation(model.representation, time_steps, embed, action, prev_state, last)
    feat = torch.cat([post.stoch, deters], -1)
    feat_dec = post.get_features()

    reconstruction_loss, i_feat = rec_loss(model.observation_decoder,
                                           feat_dec.reshape(-1, n_agents, feat_dec.shape[-1]),
                                           obs[:-1].reshape(-1, n_agents, obs.shape[-1]),
                                           1. - fake[:-1].reshape(-1, n_agents, 1))
    reward_dist = model.reward_model(feat)
    like = reward_dist.log_prob(reward[1:])
    reward_loss = - like.mean()
    reward_error = F.smooth_l1_loss(model.reward_model(feat).mode(), reward[1:])

    pcont_loss = log_prob_loss(model.pcont, feat, (1. - done[1:]))
    av_action_loss = log_prob_loss(model.av_action, feat_dec, av_action[:-1]) if av_action is not None else 0.
    i_feat = i_feat.reshape(time_steps - 1, batch_size, n_agents, -1)

    dis_loss = info_loss(i_feat[1:], model, action[1:-1], 1. - fake[1:-1].reshape(-1))
    div = state_divergence_loss(prior, post, config)

    model_loss = div + reward_loss + dis_loss + reconstruction_loss + pcont_loss + av_action_loss
    if if_log:
        wandb.log({'Model/reward_loss': reward_error, 'Model/div': div, 'Model/av_action_loss': av_action_loss,
                   'Model/reconstruction_loss': reconstruction_loss, 'Model/info_loss': dis_loss,
                   'Model/pcont_loss': pcont_loss})

    return model_loss


def sac_rollout(obs, action, last, model, actor, config):
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
    rew_states = torch.cat([items["imag_states"].stoch[:-1], items["imag_states"].deter[1:]], -1)
    raw_states = items["imag_states"].map(lambda x: x.reshape(-1, n_agents, x.shape[-1]))
    actions = items["actions"]

    with FreezeParameters([model, actor]):
        imag_reward, difference_reward = calculate_next_reward_dr(model, actions, raw_states)
        imag_reward = imag_reward.reshape(
            actions.shape[:-1]).unsqueeze(-1).mean(-2, keepdim=True).repeat(1, 1, n_agents, 1)
        discount_arr = model.pcont(rew_states).mean

        wandb.log({'Value/Max reward': imag_reward.max(), 'Value/Min reward': imag_reward.min(),
                   'Value/Reward': imag_reward.mean(), 'Value/Discount': discount_arr.mean()})

    output = [
        items["actions"][:-1].detach(),
        items["av_actions"][:-1].detach() if items["av_actions"] is not None else None,
        imag_feat[:-1].detach(),
        imag_feat[1:].detach(),
        imag_reward[:-1].detach(),
        difference_reward[:-1].detach(),
        discount_arr.detach(),
    ]

    return [batch_multi_agent(v, n_agents) for v in output]


def sac_rollout_dr(obs, action, last, model, actor, config):
    # with difference rewards
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
    rew_states = torch.cat([items["imag_states"].stoch[:-1], items["imag_states"].deter[1:]], -1)
    raw_states = items["imag_states"].map(lambda x: x.reshape(-1, n_agents, x.shape[-1]))
    actions = items["actions"]

    with FreezeParameters([model, actor]):
        imag_reward = calculate_next_reward(model, actions, raw_states)
        imag_reward = imag_reward.reshape(
            actions.shape[:-1]).unsqueeze(-1).mean(-2, keepdim=True).repeat(1, 1, n_agents, 1)
        discount_arr = model.pcont(rew_states).mean

        wandb.log({'Value/Max reward': imag_reward.max(), 'Value/Min reward': imag_reward.min(),
                   'Value/Reward': imag_reward.mean(), 'Value/Discount': discount_arr.mean()})

    output = [
        items["actions"][:-1].detach(),
        items["av_actions"][:-1].detach() if items["av_actions"] is not None else None,
        imag_feat[:-1].detach(),
        imag_feat[1:].detach(),
        imag_reward[:-1].detach(),
        discount_arr.detach(),
    ]

    return [batch_multi_agent(v, n_agents) for v in output]


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

    return [batch_multi_agent(v, n_agents) for v in output]


def critic_rollout(model, critic, states, rew_states, actions, raw_states, config):
    with FreezeParameters([model, critic]):
        imag_reward = calculate_next_reward(model, actions, raw_states)
        imag_reward = imag_reward.reshape(actions.shape[:-1]).unsqueeze(-1).mean(-2, keepdim=True)[:-1]
        value = critic(states, actions)
        discount_arr = model.pcont(rew_states).mean
        wandb.log({'Value/Max reward': imag_reward.max(), 'Value/Min reward': imag_reward.min(),
                   'Value/Reward': imag_reward.mean(), 'Value/Discount': discount_arr.mean(),
                   'Value/Value': value.mean()})

    returns = compute_return(imag_reward, value[:-1], discount_arr, bootstrap=value[-1], lmbda=config.DISCOUNT_LAMBDA,
                             gamma=config.GAMMA)
    return returns


def calculate_reward(model, states, mask=None):
    batch_size, n_agents, feat_size = states.shape
    states = states.reshape(-1, feat_size)
    reward_dist = model.reward_model(states)
    imag_reward = reward_dist.mode()
    if mask is not None:
        imag_reward *= mask

    imag_reward = imag_reward.reshape(batch_size, n_agents, 1)

    return imag_reward


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

    return rewards, (rewards - difference_rewards) * 10


def calculate_next_reward(model, actions, states):
    actions = actions.reshape(-1, actions.shape[-2], actions.shape[-1])
    next_state = model.transition(actions, states)
    imag_rew_feat = torch.cat([states.stoch, next_state.deter], -1)
    return calculate_reward(model, imag_rew_feat)


def q_rollout(obs, action, last, model, q_network, config):
    n_agents = obs.shape[2]
    with FreezeParameters([model]):
        embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
        embed = embed.reshape(obs.shape[0], obs.shape[1], n_agents, -1)
        prev_state = model.representation.initial_state(obs.shape[1], obs.shape[2], device=obs.device)
        prior, post, _ = rollout_representation(model.representation, obs.shape[0], embed, action,
                                                prev_state, last)
        post = post.map(lambda x: x.reshape((obs.shape[0] - 1) * obs.shape[1], n_agents, -1))
        items = rollout_q(model.transition, model.av_action, config.HORIZON, q_network, post)

    imag_feat = items["imag_states"].get_features()
    rew_states = torch.cat([items["imag_states"].stoch[:-1], items["imag_states"].deter[1:]], -1)
    raw_states = items["imag_states"].map(lambda x: x.reshape(-1, n_agents, x.shape[-1]))
    actions = items["actions"]

    with FreezeParameters([model, q_network]):
        imag_reward = calculate_next_reward(model, actions, raw_states)
        imag_reward = imag_reward.reshape(actions.shape[:-1]).unsqueeze(-1).mean(-2, keepdim=True)[:-1]
        discount_arr = model.pcont(rew_states).mean

        q_values = q_network.get_q_values(imag_feat, actions)

        wandb.log({'Value/Max reward': imag_reward.max(), 'Value/Min reward': imag_reward.min(),
                   'Value/Reward': imag_reward.mean(), 'Value/Discount': discount_arr.mean(),
                   'Value/Value': q_values.mean()})

    output = [
        items["actions"][:-1].detach(),
        items["av_actions"][:-1].detach() if items["av_actions"] is not None else None,
        imag_feat[:-1].detach(),
        imag_feat[1:].detach(),
        imag_reward.detach(),
        discount_arr.detach(),
    ]

    return [batch_multi_agent(v, n_agents) for v in output]


def true_q_rollout(obs, action, last, reward, model, q_network, config):
    n_agents = obs.shape[2]
    with FreezeParameters([model, q_network]):
        embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
        embed = embed.reshape(obs.shape[0], obs.shape[1], n_agents, -1)
        prev_state = model.representation.initial_state(obs.shape[1], obs.shape[2], device=obs.device)
        prior, post, _ = rollout_representation(model.representation, obs.shape[0], embed, action,
                                                prev_state, last)
        feat = post.get_features()
        discount_arr = torch.ones_like(last).to(last.device)

    output = [
        action[1:].detach(),
        None,
        feat.detach(),
        reward[1:].detach(),
        discount_arr[1:].detach(),
    ]

    return [batch_multi_agent(v, n_agents) for v in output]


def q_loss(imag_states, next_imag_states, actions, av_actions, rewards, q_network, target_q_network, discount, config):
    q_values = q_network(imag_states)
    chosen_q_values = q_values.gather(-1, actions.argmax(-1, keepdim=True))

    q_values_detach = q_values.clone().detach()
    if av_actions is not None:
        q_values_detach[av_actions == 0] = -1e10

    max_actions = q_values_detach.argmax(-1, keepdim=True)
    target_q_values = target_q_network(next_imag_states)
    target_q_values = target_q_values.gather(-1, max_actions[1:])
    target = rewards[:-1] + config.GAMMA * discount[:-1] * target_q_values

    loss = (chosen_q_values[:-1] - target.detach()).pow(2).mean()

    if np.random.randint(10) == 9:
        wandb.log({'Policy/Mean action': actions.argmax(-1).float().mean()})

    return loss.mean()


def sac_critic_loss(imag_states, next_imag_states, actions, av_actions, rewards, ac, target_ac,
                    discount, config):
    q1_values = ac.q1(imag_states)
    q1 = q1_values.gather(-1, actions.argmax(-1, keepdim=True))
    q2_values = ac.q2(imag_states)
    q2 = q2_values.gather(-1, actions.argmax(-1, keepdim=True))
    # q1_total, q2_total = q1.sum(-2, keepdim=True), q2.sum(-2, keepdim=True)

    with torch.no_grad():
        next_action, next_action_dist = target_ac.pi(next_imag_states)
        next_log_pi, next_probs_pi = next_action_dist.logits, next_action_dist.probs
        next_q1_values = target_ac.q1(next_imag_states)
        next_q2_values = target_ac.q2(next_imag_states)
        next_q_values = (
                (torch.min(next_q1_values, next_q2_values) - config.ALPHA * next_log_pi) * next_probs_pi
        ).sum(-1, keepdim=True)
        target = rewards + config.GAMMA * discount * next_q_values
        # q_total_target = rewards[:, 0:1] + config.GAMMA * discount[:, 0:1] * next_q_values.sum(-2, keepdim=True)

    loss_total_1 = (q1 - target.detach()).pow(2).mean()
    loss_total_2 = (q2 - target.detach()).pow(2).mean()
    loss_total = loss_total_1 + loss_total_2

    # loss_1 = (q1 - target.detach()).pow(2).mean()
    # loss_2 = (q2 - target.detach()).pow(2).mean()
    # loss = loss_1 + loss_2
    #
    # return loss

    return loss_total


def sac_actor_loss(imag_states, ac, config):
    action, action_dist = ac.pi(imag_states)
    log_pi, probs_pi = action_dist.logits, action_dist.probs
    q1_values = ac.q1(imag_states)
    q2_values = ac.q2(imag_states)
    q_values = torch.min(q1_values, q2_values)
    loss = ((config.ALPHA * log_pi - q_values) * probs_pi).sum(-1)

    return loss.mean()


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


def value_loss(critic, actions, imag_feat, targets):
    value_pred = critic(imag_feat, actions)
    mse_loss = (targets - value_pred) ** 2 / 2.0
    return torch.mean(mse_loss)
