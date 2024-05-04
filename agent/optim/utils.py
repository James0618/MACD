import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def rec_loss(decoder, z, x, fake):
    x_pred, feat = decoder(z)
    batch_size = np.prod(list(x.shape[:-1]))
    gen_loss1 = (F.smooth_l1_loss(x_pred, x, reduction='none', beta=0.2) * fake).sum() / batch_size
    return gen_loss1, feat


# def rew_loss(model, feat, reward, entity_mask, config):
#     batch_size = feat.shape[0]
#     model_output = model.reward_model(feat, mask=entity_mask)
#
#     reward_scale = config.REWARD_MAX - config.REWARD_MIN
#     reward = (reward.clip(config.REWARD_MIN, config.REWARD_MAX) - config.REWARD_MIN) / reward_scale
#     discrete_reward = torch.round(reward * (config.DISCRETE_CLASSES - 1)).to(torch.long)
#     target = torch.zeros((batch_size, config.DISCRETE_CLASSES)).to(reward.device).scatter_(1, discrete_reward, 1)
#
#     reward_loss = F.binary_cross_entropy_with_logits(input=model_output, target=target)
#
#     return reward_loss


def rew_loss(model, feat, reward, entity_mask, config):
    reward_dist = model.reward_model(feat, mask=entity_mask)
    like = reward_dist.log_prob(reward)

    reward_loss = - like
    max_error = abs(reward - reward_dist.mode()).max()
    return reward_loss.mean()


def ppo_loss(A, rho, eps=0.2):
    return -torch.min(rho * A, rho.clamp(1 - eps, 1 + eps) * A)


def mse(model, x, target):
    pred = model(x)
    return ((pred - target) ** 2 / 2).mean()


def entropy_loss(prob, logProb):
    return (prob * logProb).sum(-1)


def advantage(A):
    std = 1e-4 + A.std() if len(A) > 0 else 1
    adv = (A - A.mean()) / std
    adv = adv.detach()
    adv[adv != adv] = 0
    return adv


def calculate_ppo_loss(logits, rho, A, agent_mask):
    prob = F.softmax(logits, dim=-1)
    logProb = F.log_softmax(logits, dim=-1)
    polLoss = ppo_loss(A * agent_mask, rho * agent_mask)
    entLoss = entropy_loss(prob * agent_mask, logProb * agent_mask)
    return polLoss, entLoss


def batch_multi_agent(tensor, n_agents):
    return tensor.view(-1, n_agents, tensor.shape[-1]) if tensor is not None else None


def compute_return(reward, value, discount, bootstrap, lmbda, gamma):
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    target = reward + gamma * discount * next_values * (1 - lmbda)
    outputs = []
    accumulated_reward = bootstrap
    for t in reversed(range(reward.shape[0])):
        discount_factor = discount[t]
        accumulated_reward = target[t] + gamma * discount_factor * accumulated_reward * lmbda
        outputs.append(accumulated_reward)
    returns = torch.flip(torch.stack(outputs), [0])

    return returns


def info_loss(feat, model, actions, fake):
    q_feat = F.relu(model.q_features(feat))
    action_logits = model.q_action(q_feat)
    return (fake * action_information_loss(action_logits, actions)).mean()


def entity_info_loss(feat, model, actions, entity_mask, agent_mask):
    action_logits = model.q_network(feat, entity_mask)
    information_loss = action_information_loss(action_logits, actions)

    result = (agent_mask.view(-1) * information_loss).mean()

    return result


def action_information_loss(logits, target):
    criterion = nn.CrossEntropyLoss(reduction='none')
    return criterion(logits.view(-1, logits.shape[-1]), target.argmax(-1).view(-1))


def entity_log_prob_loss(model, x, mask, target):
    pred = model(x, mask)
    return -torch.mean(pred.log_prob(target))


def log_prob_loss(model, x, target):
    pred = model(x)
    return -torch.mean(pred.log_prob(target))


def kl_div_categorical(p, q):
    eps = 1e-7
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(-1)


def reshape_dist(dist, config):
    return dist.get_dist(dist.deter.shape[:-1], config.N_CATEGORICALS, config.N_CLASSES)


def state_divergence_loss(prior, posterior, config, reduce=True, balance=0.2):
    prior_dist = reshape_dist(prior, config)
    post_dist = reshape_dist(posterior, config)
    post = kl_div_categorical(post_dist, prior_dist.detach())
    pri = kl_div_categorical(post_dist.detach(), prior_dist)
    kl_div = balance * post.mean(-1) + (1 - balance) * pri.mean(-1)

    # the free bits trick of the Dreamer-V3
    # post = torch.clip(post, min=0.1)
    # pri = torch.clip(pri, min=0.1)
    # kl_div = 0.1 * post.mean(-1) + 0.5 * pri.mean(-1)

    if reduce:
        return torch.mean(kl_div)
    else:
        return kl_div
