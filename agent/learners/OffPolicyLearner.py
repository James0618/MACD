import os
import sys
import shutil
from copy import deepcopy
from pathlib import Path
import wandb
import numpy as np
import torch
import itertools

from agent.memory.DreamerMemory import DreamerMemory
from agent.models.DreamerModelV3 import DreamerModel
from agent.optim.off_loss import model_loss, actor_loss, value_loss, actor_rollout, sac_rollout, \
    sac_critic_loss, sac_actor_loss  # , q_loss
from agent.optim.utils import advantage
from environments import Env
from networks.dreamer.action import Actor
from networks.dreamer.critic import MADDPGCritic
from networks.dreamer.ac import SoftActorCritic


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
        self.ac = SoftActorCritic(config).to(config.DEVICE)
        self.target_ac = deepcopy(self.ac)

        initialize_weights(self.model, mode='xavier')
        initialize_weights(self.ac.pi)
        initialize_weights(self.ac.q1, mode='xavier')
        initialize_weights(self.ac.q2, mode='xavier')
        # self.replay_buffer = DreamerMemory(config.CAPACITY, config.SEQ_LENGTH, config.TIME_LIMIT, config.ACTION_SIZE,
        #                                    config.IN_DIM, 2, config.DEVICE, config.ENV_TYPE)
        self.replay_buffer = DreamerMemory(config.CAPACITY, config.SEQ_LENGTH, config.ACTION_SIZE,
                                           config.IN_DIM, 2, config.DEVICE, config.ENV_TYPE)
        self.entropy = config.ENTROPY
        self.step_count = -1
        self.cur_update = 1
        self.true_cur_update = 1
        self.accum_samples = 0
        self.total_samples = 0
        self.init_optimizers()
        self.n_agents = 2
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
        self.actor_optimizer = torch.optim.Adam(self.ac.pi.parameters(), lr=self.config.ACTOR_LR, weight_decay=0.00001)
        q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.q_optimizer = torch.optim.Adam(q_params, lr=self.config.Q_LR)

    def params(self):
        return {'model': {k: v.cpu() for k, v in self.model.state_dict().items()},
                'ac': {k: v.cpu() for k, v in self.ac.state_dict().items()}}

    def step(self, rollout):
        if self.n_agents != rollout['action'].shape[-2]:
            self.n_agents = rollout['action'].shape[-2]

        self.accum_samples += len(rollout['action'])
        self.total_samples += len(rollout['action'])
        self.replay_buffer.append(rollout['observation'], rollout['action'], rollout['reward'], rollout['done'],
                                  rollout['fake'], rollout['last'], rollout.get('avail_action'))
        self.step_count += 1
        if self.accum_samples < self.config.N_SAMPLES:
            return

        if len(self.replay_buffer) < self.config.MIN_BUFFER_SIZE:
            return

        self.accum_samples = 0
        sys.stdout.flush()

        # evaluate the gap between the true q and the q
        # buffer, mask, idxs = self.replay_buffer.get_full_buffer()
        # weight = self.evaluate_gap(buffer, mask)
        # self.replay_buffer.set_prob(weight=weight)

        for i in range(self.config.MODEL_EPOCHS):
            samples = self.replay_buffer.sample(self.config.MODEL_BATCH_SIZE)
            self.train_model(samples, if_log=(i == self.config.MODEL_EPOCHS - 1))

        for i in range(self.config.EPOCHS):
            samples = self.replay_buffer.sample(self.config.BATCH_SIZE)
            self.train_sac_agent(samples, if_log=(i == self.config.EPOCHS - 1))

        # for i in range(self.config.EPOCHS):
        #     samples = self.replay_buffer.sample(self.config.BATCH_SIZE, with_weights=False)
        #     self.train_true_q_agent(samples, if_log=(i == self.config.EPOCHS - 1))

        torch.save(self.params(), os.path.join(wandb.run.dir, 'model.pt'))

    # def evaluate_gap(self, buffer, mask):
    #     obs, action, last = buffer['observation'], buffer['action'], buffer['last']
    #     n_agents = obs.shape[2]
    #     time_limit, buffer_size = obs.shape[0], obs.shape[1]
    #
    #     embed = self.model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
    #     embed = embed.reshape(obs.shape[0], obs.shape[1], n_agents, -1)
    #     prev_state = self.model.representation.initial_state(obs.shape[1], obs.shape[2], device=obs.device)
    #     prior, post = true_rollout_representation(
    #         self.model.representation, obs.shape[0], embed, action, prev_state, last
    #     )
    #
    #     # post = post.map(lambda x: x.reshape(obs.shape[0] * obs.shape[1], n_agents, -1))
    #     imag_feat = post.get_features()
    #
    #     imag_q_values = self.q_network.get_q_values(
    #         imag_feat, action).sum(dim=-1).reshape(buffer_size, time_limit)
    #     true_q_values = self.true_q_network.get_q_values(
    #         imag_feat, action).sum(dim=-1).reshape(buffer_size, time_limit)
    #
    #     error = abs(imag_q_values - true_q_values)
    #     error = torch.exp(5 * error / error.max()) - 0.5
    #     weight = error[mask == 0]
    #     weight = (weight / weight.sum()).detach().cpu().numpy()
    #
    #     return weight

    def train_model(self, samples, if_log=False):
        self.model.train()
        loss = model_loss(
            self.config, self.model, samples['observation'], samples['action'], samples['av_action'],
            samples['reward'], samples['done'], samples['fake'], samples['last'], if_log
        )
        self.apply_optimizer(self.model_optimizer, self.model, loss, self.config.GRAD_CLIP)
        self.model.eval()

    def train_agent(self, samples, if_log=False):
        actions, av_actions, old_policy, imag_feat, returns = actor_rollout(
            samples['observation'], samples['action'], samples['last'], self.model, self.actor,
            self.critic if self.config.ENV_TYPE != Env.FLATLAND else self.old_critic, self.config
        )
        adv = returns.detach() - self.critic(imag_feat, actions).detach()
        if self.config.ENV_TYPE != Env.FLATLAND:
            adv = advantage(adv)
        wandb.log({'Agent/Returns': returns.mean()})
        for epoch in range(self.config.PPO_EPOCHS):
            inds = np.random.permutation(actions.shape[0])
            step = 2000
            for i in range(0, len(inds), step):
                self.cur_update += 1
                idx = inds[i:i + step]
                loss = actor_loss(imag_feat[idx], actions[idx], av_actions[idx] if av_actions is not None else None,
                                  old_policy[idx], adv[idx], self.actor, self.entropy)
                self.apply_optimizer(self.actor_optimizer, self.actor, loss, self.config.GRAD_CLIP_POLICY)
                self.entropy *= self.config.ENTROPY_ANNEALING
                val_loss = value_loss(self.critic, actions[idx], imag_feat[idx], returns[idx])
                self.apply_optimizer(self.critic_optimizer, self.critic, val_loss, self.config.GRAD_CLIP_POLICY)
                if self.config.ENV_TYPE == Env.FLATLAND and self.cur_update % self.config.TARGET_UPDATE == 0:
                    self.old_critic = deepcopy(self.critic)

        if if_log:
            wandb.log({'Agent/val_loss': val_loss, 'Agent/actor_loss': loss})

    def train_sac_agent(self, samples, if_log=False):
        actions, av_actions, imag_feat, next_imag_feat, rewards, difference_rewards, discount = sac_rollout(
            samples['observation'], samples['action'], samples['last'], self.model, self.ac.pi, self.config
        )

        for epoch in range(self.config.PPO_EPOCHS):
            inds = np.random.permutation(actions.shape[0])
            step = 512
            for i in range(0, len(inds), step):
                self.cur_update += 1
                idx = inds[i:i + step]
                q_loss = sac_critic_loss(
                    imag_feat[idx], next_imag_feat[idx], actions[idx],
                    av_actions[idx] if av_actions is not None else None,
                    difference_rewards[idx], self.ac, self.target_ac, discount[idx], config=self.config
                )
                self.apply_optimizer(self.q_optimizer, [self.ac.q1, self.ac.q2], q_loss, self.config.GRAD_CLIP_POLICY)

                pi_loss = sac_actor_loss(imag_feat[idx], self.ac, config=self.config)
                self.apply_optimizer(self.actor_optimizer, self.ac.pi, pi_loss, self.config.GRAD_CLIP_POLICY)

                with torch.no_grad():
                    for p, p_targ in zip(self.ac.parameters(), self.target_ac.parameters()):
                        p_targ.data.mul_(self.config.TAU)
                        p_targ.data.add_((1 - self.config.TAU) * p.data)

        if if_log:
            wandb.log({'Agent/q_loss': q_loss, 'Agent/pi_loss': pi_loss,
                       'Policy/Mean action': actions.argmax(-1).float().mean()})

    # def train_true_q_agent(self, samples, if_log=False):
    #     actions, av_actions, imag_feat, rewards, discount = true_q_rollout(
    #         samples['observation'], samples['action'], samples['last'], samples['reward'], self.model,
    #         self.true_q_network, self.config
    #     )
    #     next_imag_feat = imag_feat[1:]
    #     actions, imag_feat, rewards, discount = actions[:-1], imag_feat[:-1], rewards[:-1], discount[:-1]
    #
    #     for epoch in range(self.config.PPO_EPOCHS):
    #         inds = np.random.permutation(actions.shape[0])
    #         step = 512
    #         for i in range(0, len(inds), step):
    #             self.true_cur_update += 1
    #             idx = inds[i:i + step]
    #             loss = q_loss(
    #                 imag_feat[idx], next_imag_feat[idx], actions[idx],
    #                 av_actions[idx] if av_actions is not None else None,
    #                 rewards[idx], self.true_q_network, self.target_true_q_network, discount[idx], config=self.config
    #             )
    #             self.apply_optimizer(self.true_q_optimizer, self.true_q_network, loss, self.config.GRAD_CLIP_POLICY)
    #
    #             if self.true_cur_update % self.config.Q_TARGET_UPDATE == 0:
    #                 self.target_true_q_network.load_state_dict(self.true_q_network.state_dict())
    #
    #     if if_log:
    #         wandb.log({'Agent/true_q_loss': loss})

    def apply_optimizer(self, opt, model, loss, grad_clip):
        opt.zero_grad()
        loss.backward()
        if type(model) == list:
            for m in model:
                torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        opt.step()
