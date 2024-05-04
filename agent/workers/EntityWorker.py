from copy import deepcopy

import torch
from flatland.envs.agent_utils import RailAgentStatus
from collections import defaultdict

from environments import Env


class SingleWorker:

    def __init__(self, idx, env_config, controller_config, mode='simple'):
        self.runner_handle = idx
        self.env = env_config.create_env()
        self.max_num_entities = self.env.env.env.max_num_entities
        self.controller = controller_config.create_controller(mode=mode, max_num_entities=self.max_num_entities)
        self.in_dim = controller_config.IN_DIM
        self.env_type = env_config.ENV_TYPE

    def _check_handle(self, handle):
        if self.env_type == Env.STARCRAFT:
            return self.done[handle] == 0
        elif self.env_type == Env.FLATLAND:
            return self.env.agents[handle].status in (RailAgentStatus.ACTIVE, RailAgentStatus.READY_TO_DEPART) \
                   and not self.env.obs_builder.deadlock_checker.is_deadlocked(handle)
        else:
            return self.done[handle] == 0

    def _select_actions(self, state):
        state = state.unsqueeze(0)
        av_action = None
        masks = self.env.env.get_masks()
        for key in masks.keys():
            masks[key] = self._wrap(masks[key]).unsqueeze(0)

        actions = self.controller.step(
            state, av_action, agent_mask=masks['agent_mask'], entity_mask=masks['entity_mask']
        )

        return actions, state, masks, av_action

    def _wrap(self, d):
        d = torch.tensor(d).float()
        return d

    def get_absorbing_state(self):
        state = torch.zeros(1, self.in_dim)
        return state

    def _check_termination(self, info, steps_done):
        if self.env_type == Env.STARCRAFT:
            return "episode_limit" not in info
        elif self.env_type == Env.FLATLAND:
            return steps_done < self.env.max_time_steps
        else:
            return steps_done < self.env.episode_limit

    def run(self, dreamer_params):
        self.controller.receive_params(dreamer_params)

        state = self._wrap(self.env.reset())
        steps_done = 0
        episode_return = 0
        self.done = defaultdict(lambda: False)
        images = []

        while True:
            steps_done += 1
            actions, obs, masks, av_actions = self._select_actions(state)
            agent_mask = masks['agent_mask']
            entity_mask = masks['entity_mask']

            env_actions = actions[agent_mask == 1]
            next_state, reward, done, info = self.env.step(
                [action.argmax() for i, action in enumerate(env_actions)]
            )
            episode_return += reward
            next_state, reward, done = self._wrap(deepcopy(next_state)), self._wrap(deepcopy(reward)), self._wrap(deepcopy(done))
            self.done = done
            self.controller.update_buffer({"action": actions,
                                           "observation": obs,
                                           "reward": reward.reshape(1, 1),
                                           "done": done.reshape(1, 1),
                                           "agent_mask": agent_mask,
                                           "entity_mask": entity_mask,
                                           "avail_action": av_actions})

            state = next_state
            if done == 1:
                if self._check_termination(info, steps_done):
                    obs = torch.cat([self.get_absorbing_state() for i in range(self.env.n_agents)]).unsqueeze(0)
                    actions = torch.zeros(1, self.env.n_entities, actions.shape[-1])
                    index = torch.randint(0, actions.shape[-1], actions.shape[:-1], device=actions.device)
                    actions.scatter_(2, index.unsqueeze(-1), 1.)
                    items = {"observation": obs,
                             "action": actions,
                             "reward": torch.zeros(1, 1),
                             "agent_mask": torch.ones(1, self.env.n_entities),
                             "entity_mask": torch.ones(1, self.env.n_entities),
                             "done": torch.ones(1, 1),
                             "avail_action": torch.ones_like(actions) if self.env_type != Env.FLATLAND else None}
                    self.controller.update_buffer(items)
                    self.controller.update_buffer(items)
                break

        if self.env_type == Env.FLATLAND:
            reward = sum(
                [1 for agent in self.env.agents if agent.status == RailAgentStatus.DONE_REMOVED]) / self.env.n_agents
        elif self.env_type == Env.STARCRAFT:
            reward = 1. if 'battle_won' in info and info['battle_won'] else 0.
        else:
            reward = reward

        return self.controller.dispatch_buffer(), {"idx": self.runner_handle,
                                                   "reward": reward, 'return': episode_return,
                                                   "steps_done": steps_done, "images": images}
