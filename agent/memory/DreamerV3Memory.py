import numpy as np
import torch

from environments import Env


class DreamerMemory:
    def __init__(self, capacity, sequence_length, time_limit, action_size, obs_size, n_agents, device, env_type):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.time_limit = time_limit
        self.action_size = action_size
        self.obs_size = obs_size
        self.device = device
        self.env_type = env_type
        self.init_buffer(n_agents, env_type)

    def init_buffer(self, n_agents, env_type):
        self.observations = np.empty((self.capacity, n_agents, self.obs_size), dtype=np.float32)
        self.actions = np.empty((self.capacity, n_agents, self.action_size), dtype=np.float32)
        self.av_actions = np.empty((self.capacity, n_agents, self.action_size),
                                   dtype=np.float32) if env_type == Env.STARCRAFT or env_type == Env.MPE else None
        self.rewards = np.empty((self.capacity, n_agents, 1), dtype=np.float32)
        self.dones = np.empty((self.capacity, n_agents, 1), dtype=np.float32)
        self.fake = np.empty((self.capacity, n_agents, 1), dtype=np.float32)
        self.last = np.empty((self.capacity, n_agents, 1), dtype=np.float32)
        self.next_idx = 0
        self.n_agents = n_agents
        self.full = False
        self.probs = None
        self.weight = None
        self.episode_num = 0

    def append(self, obs, action, reward, done, fake, last, av_action):
        steps = len(obs)
        if self.actions.shape[-2] != action.shape[-2]:
            self.init_buffer(action.shape[-2], self.env_type)
        for i in range(len(obs)):
            self.observations[self.next_idx] = obs[i]
            self.actions[self.next_idx] = action[i]
            if av_action is not None:
                self.av_actions[self.next_idx] = av_action[i]
            self.rewards[self.next_idx] = reward[i]
            self.dones[self.next_idx] = done[i]
            self.fake[self.next_idx] = fake[i]
            self.last[self.next_idx] = last[i]
            self.next_idx = (self.next_idx + 1) % self.capacity
            self.full = self.full or self.next_idx == 0

        self.episode_num += 1
        max_weight = self.weight.max() if self.weight is not None else 1 / steps
        update_weight = max_weight * np.ones(steps)
        self.weight = np.hstack([self.weight, update_weight]) if self.weight is not None else update_weight
        self.weight = self.weight / self.weight.sum()
        self.probs = np.add.accumulate(self.weight / self.weight.sum())
        self.probs[-1] = 1.0

    def tenzorify(self, nparray):
        return torch.from_numpy(nparray).float()

    def sample(self, batch_size, with_weights=False):
        if with_weights:
            assert self.probs is not None

        return self.get_transitions(self.sample_positions(batch_size, with_weights=with_weights))

    def process_batch(self, val, idxs, batch_size):
        return torch.as_tensor(val[idxs].reshape(self.sequence_length, batch_size, self.n_agents, -1)).to(self.device)

    def get_transitions(self, idxs):
        batch_size = len(idxs)
        vec_idxs = idxs.transpose().reshape(-1)
        observation = self.process_batch(self.observations, vec_idxs, batch_size)[1:]
        reward = self.process_batch(self.rewards, vec_idxs, batch_size)[:-1]
        action = self.process_batch(self.actions, vec_idxs, batch_size)[:-1]
        av_action = self.process_batch(self.av_actions, vec_idxs, batch_size)[1:] if self.env_type == Env.STARCRAFT else None
        done = self.process_batch(self.dones, vec_idxs, batch_size)[:-1]
        fake = self.process_batch(self.fake, vec_idxs, batch_size)[1:]
        last = self.process_batch(self.last, vec_idxs, batch_size)[1:]

        return {'observation': observation, 'reward': reward, 'action': action, 'done': done,
                'fake': fake, 'last': last, 'av_action': av_action}

    def sample_position(self, with_weights=False, existing_idxs=None):
        if with_weights:
            valid_idx = False
            while not valid_idx:
                ptr = np.random.rand()
                idx = (ptr < self.probs).nonzero()[0].min()
                pre_length = self.sequence_length // 2

                post_length = self.sequence_length - pre_length
                idxs = np.arange(idx - pre_length, idx + post_length)
                # Make sure data does not cross the memory index
                valid_idx = (idxs[-1] < self.next_idx) and (idxs[0] >= 0)
                valid_idx = valid_idx and (idx not in existing_idxs)

            existing_idxs.append(idx)

        else:
            valid_idx = False
            while not valid_idx:
                idx = np.random.randint(0, self.capacity if self.full else self.next_idx - self.sequence_length)
                idxs = np.arange(idx, idx + self.sequence_length) % self.capacity
                valid_idx = self.next_idx not in idxs[1:]  # Make sure data does not cross the memory index

        return idxs

    def sample_positions(self, batch_size, with_weights=False):
        positions_list, existing_idxs = [], []
        for i in range(batch_size):
            positions_list.append(self.sample_position(with_weights=with_weights, existing_idxs=existing_idxs))

        positions = np.asarray(positions_list)

        return positions

    def __len__(self):
        return self.capacity if self.full else self.next_idx

    def clean(self):
        self.memory = list()
        self.position = 0

    def set_prob(self, weight):
        # weight: [real_buffer_size, ]
        prob = np.add.accumulate(weight / weight.sum())
        self.weight = weight
        self.probs = prob
        self.probs[-1] = 1.0

    def get_full_buffer(self):
        done_step = [-1] + self.dones[:self.next_idx].nonzero()[0].tolist()
        idxs = np.zeros((self.episode_num, self.time_limit), dtype=np.int32)
        for i in range(self.episode_num):
            episode_length = done_step[i + 1] - done_step[i]
            idxs[i, :episode_length] = np.arange(done_step[i] + 1, done_step[i + 1] + 1)

        mask = (1 - (idxs + 1).astype(np.bool)).astype(np.float32)
        vec_idxs = idxs.transpose().reshape(-1)
        observation = self.process_full_batch(self.observations, vec_idxs)
        action = self.process_full_batch(self.actions, vec_idxs)
        av_action = self.process_full_batch(self.av_actions, vec_idxs) if self.env_type == Env.STARCRAFT else None
        last = self.process_full_batch(self.last, vec_idxs)

        buffer = {
            'observation': observation, 'action': action, 'last': last, 'av_action': av_action
        }

        return buffer, mask, idxs

    def process_full_batch(self, val, idxs):
        return torch.as_tensor(val[idxs].reshape(self.time_limit, self.episode_num, self.n_agents, -1)).to(self.device)
