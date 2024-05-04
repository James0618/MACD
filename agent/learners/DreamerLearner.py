import os
import sys
import shutil
from copy import deepcopy
from pathlib import Path
import wandb
import numpy as np
import torch
import torch.nn as nn
from agent.utils.params import FreezeParameters
from networks.dreamer.old_rnns import rollout_representation

from agent.memory.DreamerMemory import DreamerMemory
from agent.models.DreamerModel import DreamerModel
from agent.optim.loss import model_loss, actor_loss, value_loss, actor_rollout, shapley_actor_rollout, \
    dae_actor_rollout, get_model_error
from agent.optim.utils import advantage
from environments import Env
from networks.dreamer.action import Actor
from networks.dreamer.critic import MADDPGCritic, DreamerV3Critic, ShapleyCritic
from agent.models.AdjacencyModel import AdjacencyModel


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

    def __init__(self, config, n_agents):
        self.config = config
        config.n_agents = n_agents
        self.n_agents = n_agents
        self.device = config.DEVICE
        self.model = DreamerModel(config).to(self.device).eval()
        self.actor = Actor(
            config.FEAT, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS,
            activation=config.activation, norm=config.norm
        ).to(self.device)
        # self.critic = MADDPGCritic(config.FEAT, config.HIDDEN).to(config.DEVICE)
        self.critic = DreamerV3Critic(
            config.FEAT, config.HIDDEN, config=config, activation=config.activation, norm=config.norm
        ).to(self.device)
        self.shapley_critic = ShapleyCritic(
            config.FEAT, config.HIDDEN, config.ACTION_SIZE, config=config, activation=config.activation,
            norm=config.norm
        ).to(self.device)
        self.adj_model = AdjacencyModel(self.config.FEAT, self.config.HIDDEN, n_agents,
                                        activation=self.config.activation, norm=self.config.norm).to(self.device)
        initialize_weights(self.model, mode='xavier')
        initialize_weights(self.adj_model, mode='xavier')
        initialize_weights(self.actor)
        initialize_weights(self.critic, mode='xavier')
        initialize_weights(self.shapley_critic, mode='xavier')
        self.old_critic = deepcopy(self.critic)
        self.replay_buffer = DreamerMemory(config.CAPACITY, config.SEQ_LENGTH, config.ACTION_SIZE, config.IN_DIM, 2,
                                           self.device, config.ENV_TYPE)
        self.entropy = config.ENTROPY
        self.step_count = -1
        self.cur_update = 1
        self.accum_samples = 0
        self.total_samples = 0

        # initialize the optimizers
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.MODEL_LR)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.ACTOR_LR, weight_decay=0.00001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.VALUE_LR)
        self.shapley_critic_optimizer = torch.optim.Adam(self.shapley_critic.parameters(), lr=self.config.VALUE_LR)
        self.adj_model_optimizer = torch.optim.Adam(self.adj_model.parameters(), lr=self.config.VALUE_LR)

        Path(config.LOG_FOLDER).mkdir(parents=True, exist_ok=True)

        wandb.init(dir=config.LOG_FOLDER, project=config.PROJECT_ID, name=config.EXP_ID)
        shutil.copyfile('configs/dreamer/DreamerLearnerConfig.py',
                        os.path.join(wandb.run.dir, 'DreamerLearnerConfig.py'))
        shutil.copyfile('configs/dreamer/DreamerAgentConfig.py',
                        os.path.join(wandb.run.dir, 'DreamerAgentConfig.py'))
        shutil.copyfile('configs/dreamer/DreamerControllerConfig.py',
                        os.path.join(wandb.run.dir, 'DreamerControllerConfig.py'))

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
                                  rollout['fake'], rollout['last'], rollout.get('avail_action'))
        self.step_count += 1
        if self.accum_samples < self.config.N_SAMPLES:
            return

        if len(self.replay_buffer) < self.config.MIN_BUFFER_SIZE:
            return

        self.accum_samples = 0
        sys.stdout.flush()

        for i in range(self.config.MODEL_EPOCHS):
            samples = self.replay_buffer.sample(self.config.MODEL_BATCH_SIZE)
            self.train_model(samples, if_log=(i == self.config.MODEL_EPOCHS - 1))
            self.train_adjacency_model(samples, if_log=(i == self.config.MODEL_EPOCHS - 1))

        for i in range(self.config.EPOCHS):
            samples = self.replay_buffer.sample(self.config.BATCH_SIZE)
            if self.config.AGENT_MODE == 'simple':
                self.train_agent(samples, if_log=(i == self.config.EPOCHS - 1))
            elif self.config.AGENT_MODE == 'shapley-critic':
                self.train_agent_with_shapley_critic(samples, if_log=(i == self.config.EPOCHS - 1))
            elif self.config.AGENT_MODE == 'shapley-rollout':
                self.train_agent_with_shapley_rollout(samples, if_log=(i == self.config.EPOCHS - 1))
            elif self.config.AGENT_MODE == 'dae':
                self.train_agent_with_dae(samples, if_log=(i == self.config.EPOCHS - 1))
            else:
                raise NotImplementedError

        if self.config.AGENT_MODE == 'shapley-critic':
            for i in range(self.config.EPOCHS):
                samples = self.replay_buffer.sample(self.config.BATCH_SIZE)
                self.train_shapley_critic(samples, if_log=(i == self.config.EPOCHS - 1))

        torch.save(self.params(), os.path.join(wandb.run.dir, 'model.pt'))

    def train_model(self, samples, if_log=False):
        self.model.train()
        loss = model_loss(
            self.config, self.model, samples['observation'], samples['action'], samples['av_action'],
            samples['reward'], samples['done'], samples['fake'], samples['last'], if_log, self.config.USE_CONSISTENCY
        )
        self.apply_optimizer(self.model_optimizer, self.model, loss, self.config.GRAD_CLIP)
        self.model.eval()

    def train_agent(self, samples, if_log=False):
        actions, av_actions, old_policy, imag_feat, returns, imag_states = actor_rollout(
            samples['observation'], samples['action'], samples['last'], self.model, self.actor,
            self.critic if self.config.ENV_TYPE != Env.FLATLAND else self.old_critic, self.config
        )
        adv = returns.detach() - self.critic(imag_feat, actions).mode().detach()
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
                val_loss, value_error = value_loss(
                    self.critic, self.old_critic, actions[idx], imag_feat[idx], returns[idx]
                )
                self.apply_optimizer(self.critic_optimizer, self.critic, val_loss, self.config.GRAD_CLIP_POLICY)
                if self.cur_update % self.config.TARGET_UPDATE == 0:
                    # self.old_critic = deepcopy(self.critic)
                    self._update_slow_target()

        if if_log:
            wandb.log({'Agent/val_loss': value_error, 'Agent/actor_loss': loss})

    def train_agent_with_shapley_critic(self, samples, if_log=False):
        actions, av_actions, old_policy, imag_feat, returns, imag_states = actor_rollout(
            samples['observation'], samples['action'], samples['last'], self.model, self.actor,
            self.critic if self.config.ENV_TYPE != Env.FLATLAND else self.old_critic, self.config,
        )
        adv = self.shapley_critic(imag_feat, actions).mode().detach()
        adv = advantage(adv)
        # adv_ = returns.detach() - self.critic(imag_feat, actions).mode().detach()
        wandb.log({'Agent/Returns': returns.mean()})        # adv = advantage(adv)
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
                val_loss, value_error = value_loss(
                    self.critic, self.old_critic, actions[idx], imag_feat[idx], returns[idx]
                )
                self.apply_optimizer(self.critic_optimizer, self.critic, val_loss, self.config.GRAD_CLIP_POLICY)
                if self.cur_update % self.config.TARGET_UPDATE == 0:
                    # self.old_critic = deepcopy(self.critic)
                    self._update_slow_target()

        if if_log:
            wandb.log({'Agent/val_loss': value_error, 'Agent/actor_loss': loss})

    def train_agent_with_shapley_rollout(self, samples, if_log=False):
        actions, av_actions, old_policy, imag_feat, returns, imag_states = actor_rollout(
            samples['observation'], samples['action'], samples['last'], self.model, self.actor,
            self.critic if self.config.ENV_TYPE != Env.FLATLAND else self.old_critic, self.config,
        )
        # adv = self.shapley_critic(imag_feat, actions).mode().detach()
        shapley_values, shapley_steps = [], 2000
        inds = np.arange(imag_states.stoch.shape[0])
        for i in range(0, len(inds), shapley_steps):
            idx = inds[i:i + shapley_steps]
            shapley_value = self.shapley_value(imag_states.map(lambda x: x[idx]), actions[idx])
            shapley_values.append(shapley_value)
        shapley_values = torch.vstack(shapley_values).detach()
        adv = advantage(shapley_values)

        wandb.log({'Agent/Returns': returns.mean()})        # adv = advantage(adv)
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
                val_loss, value_error = value_loss(
                    self.critic, self.old_critic, actions[idx], imag_feat[idx], returns[idx]
                )
                self.apply_optimizer(self.critic_optimizer, self.critic, val_loss, self.config.GRAD_CLIP_POLICY)
                if self.cur_update % self.config.TARGET_UPDATE == 0:
                    # self.old_critic = deepcopy(self.critic)
                    self._update_slow_target()

        if if_log:
            wandb.log({'Agent/val_loss': value_error, 'Agent/actor_loss': loss})

    def train_agent_with_dae(self, samples, if_log=False):
        actions, av_actions, old_policy, imag_feat, returns, advs, imag_states = dae_actor_rollout(
            samples['observation'], samples['action'], samples['last'], self.model, self.actor,
            self.critic if self.config.ENV_TYPE != Env.FLATLAND else self.old_critic, self.config,
        )
        adv = advantage(advs)
        wandb.log({'Agent/Returns': returns.mean()})        # adv = advantage(adv)
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
                val_loss, value_error = value_loss(
                    self.critic, self.old_critic, actions[idx], imag_feat[idx], returns[idx]
                )
                self.apply_optimizer(self.critic_optimizer, self.critic, val_loss, self.config.GRAD_CLIP_POLICY)
                if self.cur_update % self.config.TARGET_UPDATE == 0:
                    # self.old_critic = deepcopy(self.critic)
                    self._update_slow_target()

        if if_log:
            wandb.log({'Agent/val_loss': value_error, 'Agent/actor_loss': loss})

    def train_shapley_critic(self, samples, if_log=False):
        obs, action, last = samples['observation'], samples['action'], samples['last']
        n_agents = obs.shape[2]
        with FreezeParameters([self.model]):
            embed = self.model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
            embed = embed.reshape(obs.shape[0], obs.shape[1], n_agents, -1)
            prev_state = self.model.representation.initial_state(obs.shape[1], obs.shape[2], device=obs.device)
            _, imag_states, _ = rollout_representation(
                self.model.representation, obs.shape[0], embed, action, prev_state, last
            )
            imag_states = imag_states.map(lambda x: x.reshape((obs.shape[0] - 1) * obs.shape[1], n_agents, -1))

        action = action[:-1].reshape((obs.shape[0] - 1) * obs.shape[1], n_agents, -1)
        step = 720
        shapley_values = []
        inds = np.arange(imag_states.stoch.shape[0])
        for i in range(0, len(inds), step):
            idx = inds[i:i + step]
            shapley_value = self.shapley_value(imag_states.map(lambda x: x[idx]), action[idx])
            shapley_values.append(shapley_value)

        shapley_values = torch.vstack(shapley_values).detach()
        imag_feat = imag_states.get_features().detach()
        for i in range(40):
            value_dist = self.shapley_critic(imag_feat, action)
            loss = - value_dist.log_prob(shapley_values).mean()
            self.apply_optimizer(self.shapley_critic_optimizer, self.shapley_critic, loss, self.config.GRAD_CLIP_POLICY)

        value_error = (value_dist.mode() - shapley_values).pow(2).mean().detach()

        if if_log:
            wandb.log({'Agent/shapley_val_loss': value_error})

    def get_adjacency_matrix(self, samples):
        self.model.eval()
        steps, batch_size = samples['action'].shape[0], samples['action'].shape[1]
        agent_num = samples['action'].shape[2]
        sample_num = min(10, 2 ** (agent_num * (agent_num - 1) // 2))
        random_groups = []
        total_errors = torch.zeros(sample_num, steps - 1, batch_size)
        for i in range(sample_num):
            if_ok = False
            while not if_ok:
                random_upper_triangle = torch.triu(torch.randint(0, 2, size=(agent_num, agent_num)), diagonal=1)
                random_group = (random_upper_triangle | random_upper_triangle.t())
                if len(random_groups) == 0:
                    if_ok = True

                for group in random_groups:
                    if abs(group - random_group).sum() == 0:
                        if_ok = False
                        break
                    else:
                        if_ok = True

            random_groups.append(random_group)
            mask = random_group.unsqueeze(0).repeat(batch_size, 1, 1).to(self.device).bool()
            errors, _ = get_model_error(self.model, samples['observation'], samples['action'], samples['reward'],
                                        samples['fake'], samples['last'], mask=mask)
            total_errors[i] = errors

        min_errors = total_errors.min(dim=0)[0]
        best_mask_index = total_errors.argmin(dim=0)
        best_mask = torch.index_select(torch.stack(random_groups), 0, best_mask_index.reshape(-1))
        best_mask = best_mask.reshape(steps - 1, batch_size, agent_num, agent_num)

        original_errors, original_features = get_model_error(
            self.model, samples['observation'], samples['action'], samples['reward'], samples['fake'], samples['last'],
            mask=torch.zeros_like(mask).bool()
        )
        self.model.train()

        return best_mask, original_features

    def train_adjacency_model(self, samples, if_log=False):
        # best_matrix, original_features = self.get_adjacency_matrix(samples)
        # predicted_matrix = self.adj_model(original_features)
        # target_matrix = best_matrix.reshape(best_matrix.shape[0], best_matrix.shape[1], -1).float()
        # loss = - torch.mean(predicted_matrix.log_prob(target_matrix))
        #
        # self.apply_optimizer(self.adj_model_optimizer, self.adj_model, loss, self.config.GRAD_CLIP_POLICY)
        # if if_log:
        #     wandb.log({'Agent/adj_loss': (target_matrix - predicted_matrix.mode()).pow(2).mean().detach()})
        pass

    def shapley_value(self, imag_states, actions):
        batch_size, agent_num, feat_size = imag_states.stoch.shape
        sample_num = min(self.config.MAX_SHAPLEY_SAMPLE_NUM, 2 ** (agent_num * (agent_num - 1) // 2))
        result = torch.zeros([batch_size, agent_num, 1]).to(self.device)
        imag_states = imag_states.map(
            lambda x: x.unsqueeze(0).repeat(agent_num, 1, 1, 1).reshape(-1, agent_num, x.shape[-1])
        )
        actions = actions.unsqueeze(0).repeat(agent_num, 1, 1, 1).reshape(-1, agent_num, actions.shape[-1])
        random_groups = []
        for i in range(sample_num):
            # torch.tensor[agent_num, batch_size, agent_num]
            if_ok = False
            while not if_ok:
                random_group = torch.randint(0, 2, torch.Size([agent_num]))
                if len(random_groups) == 0:
                    random_group = torch.ones(torch.Size([agent_num]))
                    if_ok = True

                for group in random_groups:
                    if abs(group - random_group).sum() == 0:
                        if_ok = False
                        break
                    else:
                        if_ok = True

            random_groups.append(random_group)

            random_group = random_group.unsqueeze(0).unsqueeze(1).repeat(agent_num, batch_size, 1).to(self.device)
            one_hot_self_index = torch.eye(agent_num).unsqueeze(1).repeat(1, batch_size, 1).to(self.device)
            random_group_without_self = (random_group * (1 - one_hot_self_index))
            random_group_with_self = random_group_without_self + one_hot_self_index
            random_group_with_self = random_group_with_self.reshape(agent_num * batch_size, agent_num, 1)
            random_group_without_self = random_group_without_self.reshape(agent_num * batch_size, agent_num, 1)

            config = deepcopy(self.config)
            config.HORIZON = 1
            returns_with_self = shapley_actor_rollout(
                imag_states, random_group_with_self, actions, self.model, self.actor,
                self.critic if self.config.ENV_TYPE != Env.FLATLAND else self.old_critic, config
            )
            returns_without_self = shapley_actor_rollout(
                imag_states, random_group_without_self, actions, self.model, self.actor,
                self.critic if self.config.ENV_TYPE != Env.FLATLAND else self.old_critic, config
            )
            differences = returns_with_self - returns_without_self
            differences = differences.reshape(agent_num, batch_size, agent_num, 1)

            temp = torch.zeros([batch_size, agent_num, 1]).to(self.device)
            for i in range(agent_num):
                temp[:, i] = differences[i, :, i]

            result += temp

        return result / sample_num

    def apply_optimizer(self, opt, model, loss, grad_clip):
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

    def _update_slow_target(self):
        mix = self.config.slow_target_fraction
        for s, d in zip(self.critic.parameters(), self.old_critic.parameters()):
            d.data = mix * s.data + (1 - mix) * d.data
