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
from agent.optim.loss import model_loss, central_model_loss, continuous_actor_loss, value_loss, \
    continuous_actor_rollout, shapley_actor_rollout, dae_actor_rollout, get_model_error, ac_actor_loss, ac_value_loss
from agent.optim.utils import advantage
from environments import Env
from networks.dreamer.action import ContinuousActor
from networks.dreamer.critic import ShapleyCritic, DreamerV3Critic


class RewardEMA(object):
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.values = torch.zeros((2,)).to(device)
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        self.values = self.alpha * x_quantile + (1 - self.alpha) * self.values
        scale = torch.clip(self.values[1] - self.values[0], min=1.0)
        offset = self.values[0]

        return offset.detach(), scale.detach()


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


class ContinuousLearner:
    def __init__(self, config, n_agents):
        self.config = config
        config.n_agents = n_agents
        self.n_agents = n_agents
        self.device = config.DEVICE
        self.model = DreamerModel(config).to(self.device).eval()
        self.actor = ContinuousActor(
            config.FEAT, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS,
            activation=config.activation, norm=config.norm
        ).to(self.device)
        # self.critic = DreamerV3Critic(
        #     config.FEAT, config.HIDDEN, config=config, activation=config.activation, norm=config.norm
        # ).to(self.device)

        if self.config.AGENT_MODE == 'shapley-rollout-central':
            self.critic = ShapleyCritic(
                config.FEAT, config.HIDDEN, config=config, activation=config.activation, norm=config.norm
            ).to(self.device)
        elif self.config.AGENT_MODE == 'shapley-rollout' or self.config.AGENT_MODE == 'simple':
            self.critic = DreamerV3Critic(
                config.FEAT, config.HIDDEN, config=config, activation=config.activation, norm=config.norm
            ).to(self.device)
        else:
            raise NotImplementedError

        initialize_weights(self.model, mode='xavier')
        initialize_weights(self.actor)
        initialize_weights(self.critic, mode='xavier')
        self.old_critic = deepcopy(self.critic)
        self.replay_buffer = DreamerMemory(config.CAPACITY, config.SEQ_LENGTH, config.ACTION_SIZE, config.IN_DIM, 2,
                                           self.device, config.ENV_TYPE, central_obs_size=config.CENTRAL_OBS_SIZE)
        self.reward_ema = RewardEMA(device=self.device)
        self.entropy = config.ENTROPY
        self.step_count = -1
        self.cur_update = 1
        self.accum_samples = 0
        self.total_samples = 0

        # initialize the optimizers
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.MODEL_LR)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.ACTOR_LR, weight_decay=0.00001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.VALUE_LR)

        Path(config.LOG_FOLDER).mkdir(parents=True, exist_ok=True)

        wandb.init(dir=config.LOG_FOLDER, project=config.PROJECT_ID, name=config.EXP_ID, notes=config.NOTES)
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

    def load_params(self, params):
        self.model.load_state_dict(params['model'])
        self.actor.load_state_dict(params['actor'])
        self.critic.load_state_dict(params['critic'])

    def step(self, rollout):
        if self.n_agents != rollout['action'].shape[-2]:
            self.n_agents = rollout['action'].shape[-2]

        self.accum_samples += len(rollout['action'])
        self.total_samples += len(rollout['action'])
        self.replay_buffer.append(
            rollout['observation'], rollout['action'], rollout['reward'], rollout['done'], rollout['fake'],
            rollout['last'], rollout.get('avail_action'), central_obs=rollout['central_observation']
        )
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

        for i in range(self.config.EPOCHS):
            samples = self.replay_buffer.sample(self.config.BATCH_SIZE)
            if self.config.AGENT_MODE == 'simple':
                self.train_agent(samples, if_log=(i == self.config.EPOCHS - 1))
            elif self.config.AGENT_MODE == 'shapley-rollout':
                self.train_agent_with_shapley_rollout(samples, if_log=(i == self.config.EPOCHS - 1))
            elif self.config.AGENT_MODE == 'shapley-rollout-central':
                self.train_agent_with_shapley_rollout_central(samples, if_log=(i == self.config.EPOCHS - 1))
            elif self.config.AGENT_MODE == 'dae':
                self.train_agent_with_dae(samples, if_log=(i == self.config.EPOCHS - 1))
            else:
                raise NotImplementedError

        # self.entropy *= self.config.ENTROPY_ANNEALING
        torch.save(self.params(), os.path.join(wandb.run.dir, 'model.pt'))

    def step_ac(self, rollout):
        if self.n_agents != rollout['action'].shape[-2]:
            self.n_agents = rollout['action'].shape[-2]

        self.accum_samples += len(rollout['action'])
        self.total_samples += len(rollout['action'])
        self.replay_buffer.append(
            rollout['observation'], rollout['action'], rollout['reward'], rollout['done'], rollout['fake'],
            rollout['last'], rollout.get('avail_action'), central_obs=rollout['central_observation']
        )
        self.step_count += 1
        if self.accum_samples < self.config.N_SAMPLES:
            return

        if len(self.replay_buffer) < self.config.MIN_BUFFER_SIZE:
            return

        self.accum_samples = 0
        sys.stdout.flush()

        # train the world model
        for i in range(self.config.MODEL_EPOCHS):
            samples = self.replay_buffer.sample(self.config.MODEL_BATCH_SIZE)
            self.train_model(samples, if_log=True)

        # train the actor and critic
        for i in range(self.config.EPOCHS):
            samples = self.replay_buffer.sample(self.config.BATCH_SIZE)
            self.train_agent_ac(samples, if_log=True)

        torch.save(self.params(), os.path.join(wandb.run.dir, 'model.pt'))

    def train_model(self, samples, if_log=False):
        self.model.train()
        if self.config.RECONSTRUCT_CENTRAL:
            loss = central_model_loss(
                self.config, self.model, samples['observation'], samples['central_observation'], samples['action'],
                samples['av_action'], samples['reward'], samples['done'], samples['fake'], samples['last'], if_log,
                self.config.USE_CONSISTENCY
            )
        else:
            loss = model_loss(
                self.config, self.model, samples['observation'], samples['action'], samples['av_action'],
                samples['reward'], samples['done'], samples['fake'], samples['last'], if_log,
                self.config.USE_CONSISTENCY
            )
        self.apply_optimizer(self.model_optimizer, self.model, loss, self.config.GRAD_CLIP)
        self.model.eval()

    def train_agent(self, samples, if_log=False):
        actions, av_actions, old_policy, imag_feat, returns, imag_states = continuous_actor_rollout(
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
                loss = continuous_actor_loss(
                    imag_feat[idx], actions[idx], av_actions[idx] if av_actions is not None else None,
                    old_policy[idx], adv[idx], self.actor, self.entropy, self.config.PPO_CLIP_COEF
                )
                self.apply_optimizer(self.actor_optimizer, self.actor, loss, self.config.GRAD_CLIP_POLICY)
                # self.entropy *= self.config.ENTROPY_ANNEALING
                val_loss, value_error = value_loss(
                    self.critic, self.old_critic, actions[idx], imag_feat[idx], returns[idx]
                )
                self.apply_optimizer(self.critic_optimizer, self.critic, val_loss, self.config.GRAD_CLIP_POLICY)
                if self.cur_update % self.config.TARGET_UPDATE == 0:
                    # self.old_critic = deepcopy(self.critic)
                    self._update_slow_target()

        if if_log:
            wandb.log({'Agent/val_loss': value_error, 'Agent/actor_loss': loss})

    def train_agent_ac(self, samples, if_log=False):
        actions, av_actions, old_policy, imag_feat, returns, imag_states = continuous_actor_rollout(
            samples['observation'], samples['action'], samples['last'], self.model, self.actor,
            self.critic if self.config.ENV_TYPE != Env.FLATLAND else self.old_critic, self.config
        )
        value = self.critic(imag_feat, actions).mode().detach()
        offset, scale = self.reward_ema(returns)
        normed_target = (returns - offset) / scale
        normed_base = (value - offset) / scale
        adv = normed_target - normed_base

        wandb.log({'Agent/Returns': returns.mean()})
        for epoch in range(self.config.PPO_EPOCHS):
            actor_loss = ac_actor_loss(
                imag_feat, actions, av_actions if av_actions is not None else None,
                adv, self.actor, self.entropy
            )
            self.apply_optimizer(self.actor_optimizer, self.actor, actor_loss, self.config.GRAD_CLIP_POLICY)
            # self.entropy *= self.config.ENTROPY_ANNEALING
            val_loss, value_error = ac_value_loss(
                self.critic, self.old_critic, actions, imag_feat, returns
            )
            self.apply_optimizer(self.critic_optimizer, self.critic, val_loss, self.config.GRAD_CLIP_POLICY)
            if self.cur_update % self.config.TARGET_UPDATE == 0:
                # self.old_critic = deepcopy(self.critic)
                self._update_slow_target()

        if if_log:
            wandb.log({'Agent/val_loss': value_error, 'Agent/actor_loss': actor_loss})

    def train_agent_with_shapley_rollout(self, samples, if_log=False):
        actions, av_actions, old_policy, imag_feat, returns, imag_states = continuous_actor_rollout(
            samples['observation'], samples['action'], samples['last'], self.model, self.actor,
            self.critic if self.config.ENV_TYPE != Env.FLATLAND else self.old_critic, self.config,
        )
        # shapley_values, shapley_steps = [], 2000
        # inds = np.arange(imag_states.stoch.shape[0])
        # for i in range(0, len(inds), shapley_steps):
        #     idx = inds[i:i + shapley_steps]
        #     shapley_value = self.shapley_value(imag_states.map(lambda x: x[idx]), actions[idx])
        #     shapley_values.append(shapley_value)
        # shapley_values = torch.vstack(shapley_values).detach()
        # adv = advantage(shapley_values)

        wandb.log({'Agent/Returns': returns.mean()})        # adv = advantage(adv)
        for epoch in range(self.config.PPO_EPOCHS):
            inds = np.random.permutation(actions.shape[0])
            step = 2000
            for i in range(0, len(inds), step):
                self.cur_update += 1
                idx = inds[i:i + step]
                shapley_value = self.shapley_value(imag_states.map(lambda x: x[idx]), actions[idx])
                adv = advantage(shapley_value)
                loss = continuous_actor_loss(
                    imag_feat[idx], actions[idx], av_actions[idx] if av_actions is not None else None,
                    old_policy[idx], adv, self.actor, self.entropy
                )

                # loss = continuous_actor_loss(
                #     imag_feat[idx], actions[idx], av_actions[idx] if av_actions is not None else None,
                #     old_policy[idx], adv[idx], self.actor, self.entropy
                # )
                self.apply_optimizer(self.actor_optimizer, self.actor, loss, self.config.GRAD_CLIP_POLICY)
                # self.entropy *= self.config.ENTROPY_ANNEALING
                val_loss, value_error = value_loss(
                    self.critic, self.old_critic, actions[idx], imag_feat[idx], returns[idx]
                )
                self.apply_optimizer(self.critic_optimizer, self.critic, val_loss, self.config.GRAD_CLIP_POLICY)
                if self.cur_update % self.config.TARGET_UPDATE == 0:
                    # self.old_critic = deepcopy(self.critic)
                    self._update_slow_target()

        if if_log:
            wandb.log({'Agent/val_loss': value_error, 'Agent/actor_loss': loss})

    def train_agent_with_shapley_rollout_central(self, samples, if_log=False):
        actions, av_actions, old_policy, imag_feat, returns, imag_states = continuous_actor_rollout(
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
                loss = continuous_actor_loss(
                    imag_feat[idx], actions[idx], av_actions[idx] if av_actions is not None else None,
                    old_policy[idx], adv[idx], self.actor, self.entropy
                )
                self.apply_optimizer(self.actor_optimizer, self.actor, loss, self.config.GRAD_CLIP_POLICY)
                value_dist = self.critic(imag_feat[idx], actions[idx], central=True)
                val_loss = - value_dist.log_prob(returns[idx][:, :1])
                val_loss, value_error = val_loss.mean(), (value_dist.mode() - returns[idx][:, :1]).pow(2).detach()

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
                loss = continuous_actor_loss(
                    imag_feat[idx], actions[idx], av_actions[idx] if av_actions is not None else None,
                    old_policy[idx], adv[idx], self.actor, self.entropy
                )
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
            config.HORIZON = self.config.SHAPLEY_HORIZON
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
