import os
import sys
import shutil
from copy import deepcopy
from pathlib import Path

import wandb
import torch
import numpy as np

from agent.memory.EntityMemory import DreamerMemory
from agent.models.TransfDreamer import DreamerModel
from agent.optim.entity_loss import model_loss, actor_loss, value_loss, actor_rollout, q_network_rollout
from agent.optim.utils import advantage
from networks.dreamer.utils import RewardEMA
from environments import Env
from networks.dreamer.action import Actor, EntityActor
from networks.dreamer.critic import MADDPGCritic, EntityPPOCritic


def orthogonal_init(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


def initialize_weights(mod, scale=1.0, mode='ortho'):
    for p in mod.parameters():
        if mode == 'ortho':
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
        elif mode == 'xavier':
            if len(p.data.shape) >= 2:
                torch.nn.init.xavier_uniform_(p.data)


class DreamerLearner:
    def __init__(self, config):
        self.config = config
        self.model = DreamerModel(config).to(config.DEVICE).eval()
        self.actor = EntityActor(
            config.FEAT, config.ACTION_SIZE, config.ACTION_HIDDEN, config.MAX_NUM_ENTITIES
        ).to(config.DEVICE)
        self.critic = EntityPPOCritic(config.FEAT, config.HIDDEN).to(config.DEVICE)
        self.target_q_network = deepcopy(self.q_network)

        initialize_weights(self.model, mode='xavier')
        initialize_weights(self.actor)
        initialize_weights(self.critic, mode='xavier')
        self.old_critic = deepcopy(self.critic)
        self.replay_buffer = DreamerMemory(config.CAPACITY, config.SEQ_LENGTH, config.ACTION_SIZE, config.IN_DIM, 2,
                                           config.DEVICE, config.ENV_TYPE)
        self.entropy = config.ENTROPY
        self.step_count = -1
        self.cur_update = 1
        self.accum_samples = 0
        self.total_samples = 0
        self.init_optimizers()
        self.n_agents = 2
        self.model_update_index = 0

        self.reward_ema = RewardEMA(device=config.DEVICE)

        Path(config.LOG_FOLDER).mkdir(parents=True, exist_ok=True)

        wandb.init(dir=config.LOG_FOLDER, project='mamba', name=config.EXP_ID)
        shutil.copyfile('configs/dreamer/DreamerLearnerConfig.py',
                        os.path.join(wandb.run.dir, 'DreamerLearnerConfig.py'))
        shutil.copyfile('configs/dreamer/DreamerAgentConfig.py',
                        os.path.join(wandb.run.dir, 'DreamerAgentConfig.py'))
        shutil.copyfile('configs/dreamer/DreamerControllerConfig.py',
                        os.path.join(wandb.run.dir, 'DreamerControllerConfig.py'))

    def init_optimizers(self):
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.MODEL_LR)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.ACTOR_LR, weight_decay=0.00001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.VALUE_LR)

    def params(self):
        return {'model': {k: v.cpu() for k, v in self.model.state_dict().items()},
                'actor': {k: v.cpu() for k, v in self.actor.state_dict().items()},
                'critic': {k: v.cpu() for k, v in self.critic.state_dict().items()}}

    def step(self, rollout):
        if self.n_agents != rollout['action'].shape[-2]:
            self.n_agents = rollout['action'].shape[-2]

        self.accum_samples += len(rollout['action'])
        self.total_samples += len(rollout['action'])
        self.replay_buffer.append(rollout['observation'], rollout['action'], rollout['reward'], rollout['done'],
                                  rollout['agent_mask'], rollout['entity_mask'], rollout['last'],
                                  rollout.get('avail_action'))
        self.step_count += 1
        if self.accum_samples < self.config.N_SAMPLES:
            return

        if len(self.replay_buffer) < self.config.MIN_BUFFER_SIZE:
            return

        self.accum_samples = 0
        sys.stdout.flush()

        for i in range(self.config.MODEL_EPOCHS):
            samples = self.replay_buffer.sample(self.config.MODEL_BATCH_SIZE)
            loss_info = self.train_model(samples)
            self.model_update_index += 1

            wandb.log({'Model/reward_loss': loss_info['reward_loss'],
                       'Model/reconstruction_loss': loss_info['reconstruction_loss'],
                       'Model/info_loss': loss_info['info_loss'],
                       'Model/pcont_loss': loss_info['pcont_loss'],
                       'Model/div': loss_info['div'], 'Model/index': self.model_update_index})

        for i in range(self.config.EPOCHS):
            samples = self.replay_buffer.sample(self.config.BATCH_SIZE)
            # self.train_agent(samples, if_log=(i == self.config.EPOCHS - 1))
            self.train_q_agent(samples, if_log=(i == self.config.EPOCHS - 1))

        torch.save(self.params(), os.path.join(wandb.run.dir, 'model.pt'))

    def train_model(self, samples):
        self.model.train()
        loss, loss_info = model_loss(
            self.config, self.model, samples['observation'], samples['action'], samples['av_action'],
            samples['reward'], samples['done'], samples['agent_mask'], samples['entity_mask'], samples['last']
        )
        self.apply_optimizer(self.model_optimizer, self.model, loss, self.config.GRAD_CLIP)
        self.model.eval()

        return loss_info

    def train_agent(self, samples, if_log=False):
        actions, av_actions, old_policy, imag_feat, returns, entity_mask, agent_mask = actor_rollout(
            samples['observation'], samples['action'], samples['last'], samples['entity_mask'], samples['agent_mask'],
            self.model, self.actor, self.critic if self.config.ENV_TYPE != Env.FLATLAND else self.old_critic,
            self.config
        )
        adv = returns.detach() - self.critic(imag_feat, entity_mask, agent_mask).detach()
        # _, scale = self.reward_ema(returns.detach())
        # adv = adv / scale

        wandb.log({'Agent/Returns': returns.mean()})
        for epoch in range(self.config.PPO_EPOCHS):
            inds = np.random.permutation(actions.shape[0])
            step = 2000
            for i in range(0, len(inds), step):
                self.cur_update += 1
                idx = inds[i:i + step]
                loss = actor_loss(imag_feat[idx], actions[idx], av_actions[idx] if av_actions is not None else None,
                                  old_policy[idx], adv[idx], self.actor, self.entropy, entity_mask[idx],
                                  agent_mask[idx])
                self.apply_optimizer(self.actor_optimizer, self.actor, loss, self.config.GRAD_CLIP_POLICY)
                self.entropy *= self.config.ENTROPY_ANNEALING
                val_loss = value_loss(self.critic, actions[idx], imag_feat[idx], returns[idx], entity_mask[idx],
                                      agent_mask[idx])
                self.apply_optimizer(self.critic_optimizer, self.critic, val_loss, self.config.GRAD_CLIP_POLICY)
                if self.config.ENV_TYPE == Env.FLATLAND and self.cur_update % self.config.TARGET_UPDATE == 0:
                    self.old_critic = deepcopy(self.critic)

        if if_log:
            wandb.log({'Agent/val_loss': val_loss, 'Agent/actor_loss': loss})

    def train_q_agent(self, samples, if_log=False):
        actions, av_actions, old_policy, imag_feat, returns, entity_mask, agent_mask = q_network_rollout(
            samples['observation'], samples['action'], samples['last'], samples['entity_mask'], samples['agent_mask'],
            self.model, self.q_network, self.target_q_network, self.config
        )
        adv = returns.detach() - self.critic(imag_feat, entity_mask, agent_mask).detach()
        # _, scale = self.reward_ema(returns.detach())
        # adv = adv / scale

        wandb.log({'Agent/Returns': returns.mean()})
        for epoch in range(self.config.PPO_EPOCHS):
            inds = np.random.permutation(actions.shape[0])
            step = 2000
            for i in range(0, len(inds), step):
                self.cur_update += 1
                idx = inds[i:i + step]
                loss = actor_loss(imag_feat[idx], actions[idx], av_actions[idx] if av_actions is not None else None,
                                  old_policy[idx], adv[idx], self.actor, self.entropy, entity_mask[idx],
                                  agent_mask[idx])
                self.apply_optimizer(self.actor_optimizer, self.actor, loss, self.config.GRAD_CLIP_POLICY)
                self.entropy *= self.config.ENTROPY_ANNEALING
                val_loss = value_loss(self.critic, actions[idx], imag_feat[idx], returns[idx], entity_mask[idx],
                                      agent_mask[idx])
                self.apply_optimizer(self.critic_optimizer, self.critic, val_loss, self.config.GRAD_CLIP_POLICY)
                if self.config.ENV_TYPE == Env.FLATLAND and self.cur_update % self.config.TARGET_UPDATE == 0:
                    self.old_critic = deepcopy(self.critic)

        if if_log:
            wandb.log({'Agent/val_loss': val_loss, 'Agent/actor_loss': loss})

    def apply_optimizer(self, opt, model, loss, grad_clip):
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
