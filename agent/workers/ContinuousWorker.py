from copy import deepcopy

import torch
from flatland.envs.agent_utils import RailAgentStatus
from collections import defaultdict

from environments import Env


class ContinuousWorker:
    def __init__(self, idx, env_config, controller_config, mode='simple'):
        self.runner_handle = idx
        self.env = env_config.create_env()
        self.n_agents = self.env.env.n_agents
        self.config = controller_config
        self.controller = controller_config.create_controller(mode=mode)
        self.in_dim = controller_config.IN_DIM
        self.central_obs_size = controller_config.CENTRAL_OBS_SIZE
        self.env_type = env_config.ENV_TYPE

    def _check_handle(self, handle):
        if self.env_type == Env.STARCRAFT:
            return self.done[handle] == 0
        elif self.env_type == Env.FLATLAND:
            return self.env.agents[handle].status in (RailAgentStatus.ACTIVE, RailAgentStatus.READY_TO_DEPART) \
                   and not self.env.obs_builder.deadlock_checker.is_deadlocked(handle)
        else:
            return self.done[handle] == 0

    def _select_actions(self, state, central_state=None, deterministic=False):
        avail_actions = []
        observations = []
        central_observations = []
        fakes = []
        if self.env_type == Env.FLATLAND:
            nn_mask = (1. - torch.eye(self.env.n_agents)).bool()
        else:
            nn_mask = None

        for handle in range(self.env.n_agents):
            if self.env_type == Env.FLATLAND:
                for opp_handle in self.env.obs_builder.encountered[handle]:
                    if opp_handle != -1:
                        nn_mask[handle, opp_handle] = False
            elif self.env_type == Env.MUJOCO:
                avail_actions = []
            else:
                avail_actions.append(torch.tensor(self.env.get_avail_agent_actions(handle)))

            if self._check_handle(handle) and handle in state:
                fakes.append(torch.zeros(1, 1))
                observations.append(state[handle].unsqueeze(0))
            elif self.done[handle] == 1:
                fakes.append(torch.ones(1, 1))
                observations.append(self.get_absorbing_state())
            else:
                fakes.append(torch.zeros(1, 1))
                obs = torch.tensor(self.env.obs_builder._get_internal(handle)).float().unsqueeze(0)
                observations.append(obs)

            if central_state is not None:
                if self._check_handle(handle) and handle in central_state:
                    central_observations.append(central_state[handle].unsqueeze(0))
                elif self.done[handle] == 1:
                    central_observations.append(self.get_absorbing_state())
                else:
                    obs = torch.tensor(self.env.obs_builder._get_internal(handle)).float().unsqueeze(0)
                    central_observations.append(obs)

        observations = torch.cat(observations).unsqueeze(0)
        central_observations = torch.cat(central_observations).unsqueeze(0) if central_state is not None else None
        av_action = torch.stack(avail_actions).unsqueeze(0) if len(avail_actions) > 0 else None
        nn_mask = nn_mask.unsqueeze(0).repeat(8, 1, 1) if nn_mask is not None else None
        actions = self.controller.step(observations, av_action, nn_mask, deterministic=deterministic)

        return actions, observations, central_observations, torch.cat(fakes).unsqueeze(0), av_action

    def _wrap(self, d):
        for key, value in d.items():
            d[key] = torch.tensor(value).float()
        return d

    def _mujoco_wrap(self, d):
        result = {}
        for i, state in enumerate(d):
            result[i] = torch.tensor(state).float()
        return result

    def get_absorbing_state(self, if_central=False):
        if if_central:
            return torch.zeros(1, self.central_obs_size)
        else:
            return torch.zeros(1, self.in_dim)

    def augment(self, data, inverse=False):
        aug = []
        default = list(data.values())[0].reshape(1, -1)
        for handle in range(self.env.n_agents):
            if handle in data.keys():
                aug.append(data[handle].reshape(1, -1))
            else:
                aug.append(torch.ones_like(default) if inverse else torch.zeros_like(default))
        return torch.cat(aug).unsqueeze(0)

    def _check_termination(self, info, steps_done):
        if self.env_type == Env.STARCRAFT:
            return "episode_limit" not in info
        elif self.env_type == Env.FLATLAND:
            return steps_done < self.env.max_time_steps
        else:
            return steps_done < self.env.episode_limit

    def run(self, dreamer_params):
        self.controller.receive_params(dreamer_params)

        if self.config.MULTI_TASK:
            _, _, _ = self.env.reset_task_params()

        state, central_state, _ = self.env.reset()
        self.controller.init_rnns()
        state = self._mujoco_wrap(state)
        central_state = self._mujoco_wrap(central_state)
        steps_done = 0
        episode_return = 0
        self.done = defaultdict(lambda: False)

        while True:
            steps_done += 1
            actions, obs, central_obs, fakes, av_actions = self._select_actions(
                state, central_state=central_state, deterministic=True
            )
            next_state, next_central_state, rewards, dones, infos, _ = self.env.step(actions[0])
            next_state = self._mujoco_wrap(deepcopy(next_state))
            next_central_state = self._mujoco_wrap(deepcopy(next_central_state))
            done, reward = self._mujoco_wrap(deepcopy(dones)), self._mujoco_wrap(deepcopy(rewards))

            episode_return += reward[0]
            self.done = done
            self.controller.update_buffer({"action": actions,
                                           "observation": obs,
                                           "central_observation": central_obs,
                                           "reward": self.augment(reward),
                                           "done": self.augment(done),
                                           "fake": fakes,
                                           "avail_action": av_actions})
            state = next_state
            central_state = next_central_state
            if all([done[key] == 1 for key in range(self.env.n_agents)]):
                if not infos[0]['bad_transition']:
                    obs = torch.cat([self.get_absorbing_state() for i in range(self.env.n_agents)]).unsqueeze(0)
                    actions = torch.zeros(1, self.env.n_agents, actions.shape[-1])
                    items = {"observation": obs,
                             "central_observation": central_obs,
                             "action": actions,
                             "reward": torch.zeros(1, self.env.n_agents, 1),
                             "fake": torch.ones(1, self.env.n_agents, 1),
                             "done": torch.ones(1, self.env.n_agents, 1),
                             "avail_action": torch.ones_like(actions) if self.env_type == Env.STARCRAFT else None}
                    self.controller.update_buffer(items)
                    self.controller.update_buffer(items)
                break

        return self.controller.dispatch_buffer(), {
            "idx": self.runner_handle, 'return': episode_return, "steps_done": steps_done
        }

    def run_ac(self, dreamer_params, start=False, last_state=None, last_central_state=None, episode_return=0,
               steps_done=0):
        self.controller.receive_params(dreamer_params)

        if start:
            state, central_state, _ = self.env.reset()
            self.controller.init_rnns()
            state = self._mujoco_wrap(state)
            central_state = self._mujoco_wrap(central_state)
            episode_return, steps_done = 0, 0
            self.done = defaultdict(lambda: False)
        else:
            assert last_state is not None
            assert last_central_state is not None
            state = last_state
            central_state = last_central_state
            episode_return, steps_done = episode_return, steps_done

        for _ in range(self.config.TRAIN_RATIO):
            steps_done += 1
            actions, obs, central_obs, fakes, av_actions = self._select_actions(
                state, central_state=central_state, deterministic=True
            )
            next_state, next_central_state, rewards, dones, infos, _ = self.env.step(actions[0])
            next_state = self._mujoco_wrap(deepcopy(next_state))
            next_central_state = self._mujoco_wrap(deepcopy(next_central_state))
            done, reward = self._mujoco_wrap(deepcopy(dones)), self._mujoco_wrap(deepcopy(rewards))

            episode_return += reward[0]
            self.done = done
            self.controller.update_buffer({"action": actions,
                                           "observation": obs,
                                           "central_observation": central_obs,
                                           "reward": self.augment(reward),
                                           "done": self.augment(done),
                                           "fake": fakes,
                                           "avail_action": av_actions})
            state = next_state
            central_state = next_central_state

            done_flag = all([done[key] == 1 for key in range(self.env.n_agents)])
            if done_flag:
                if not infos[0]['bad_transition']:
                    obs = torch.cat([self.get_absorbing_state() for i in range(self.env.n_agents)]).unsqueeze(0)
                    actions = torch.zeros(1, self.env.n_agents, actions.shape[-1])
                    index = torch.randint(0, actions.shape[-1], actions.shape[:-1], device=actions.device)
                    actions.scatter_(2, index.unsqueeze(-1), 1.)
                    items = {"observation": obs,
                             "action": actions,
                             "reward": torch.zeros(1, self.env.n_agents, 1),
                             "fake": torch.ones(1, self.env.n_agents, 1),
                             "done": torch.ones(1, self.env.n_agents, 1),
                             "avail_action": torch.ones_like(actions) if self.env_type == Env.STARCRAFT else None}
                    self.controller.update_buffer(items)
                    self.controller.update_buffer(items)
                break

        return self.controller.dispatch_buffer_ac(done_flag), {
            "idx": self.runner_handle, 'return': episode_return, "steps_done": steps_done, 'done_flag': done_flag,
            'state': state, 'central_state': central_state
        }

    def evaluate(self, dreamer_params, test_num=4):
        self.controller.receive_params(dreamer_params)

        average_return, average_step = 0, 0
        for i in range(test_num):
            state, central_state, _ = self.env.reset()
            self.controller.init_rnns()
            state = self._mujoco_wrap(state)
            central_state = self._mujoco_wrap(central_state)
            steps_done, episode_return = 0, 0
            self.done = defaultdict(lambda: False)

            while True:
                steps_done += 1
                actions, obs, central_obs, fakes, av_actions = self._select_actions(
                    state, central_state=central_state, deterministic=True
                )
                next_state, next_central_state, rewards, dones, infos, _ = self.env.step(actions[0])
                next_state = self._mujoco_wrap(deepcopy(next_state))
                next_central_state = self._mujoco_wrap(deepcopy(next_central_state))
                done, reward = self._mujoco_wrap(deepcopy(dones)), self._mujoco_wrap(deepcopy(rewards))

                episode_return += reward[0]
                self.done = done
                state = next_state
                central_state = next_central_state
                if all([done[key] == 1 for key in range(self.env.n_agents)]):
                    break

            average_return += episode_return
            average_step += steps_done

        return {"idx": self.runner_handle, 'return': average_return / test_num, "steps_done": average_step / test_num}
