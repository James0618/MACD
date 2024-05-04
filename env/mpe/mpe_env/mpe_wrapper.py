import importlib
from .environment import MultiAgentEnv
import gym
import copy
from gym import ObservationWrapper, spaces
from gym.spaces import flatdim
import numpy as np
from gym.wrappers import TimeLimit as GymTimeLimit
from .task import *


try:
    from .animate.plotly_animator import MPEAnimator
except:
    from .animate.pyplot_animator import MPEAnimator

    print('Using matplotlib to save the animation because plolty is not available')


class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
                self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not all(done)
            done = len(observation) * [True]
        return observation, reward, done, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple(
            [
                spaces.flatten(obs_space, obs)
                for obs_space, obs in zip(self.env.observation_space, observation)
            ]
        )


class MPEWrapper(MultiAgentEnv):
    def __init__(self, key, time_limit, **kwargs):
        self.original_task = Task()
        self.task_name = 'simple_spread'
        self.episode_limit = time_limit
        self.env_dict = {}
        for key in TaskNames:
            self.env_dict[key] = self.get_env(key, time_limit, kwargs)

        # basic env variables
        self.max_num_agents = self.original_task.max_num_agents
        self.max_num_landmarks = self.original_task.max_num_landmarks
        self.max_num_entities = self.original_task.max_num_entities

        self.n_entity_types = 3
        self._obs = None
        self._state = None

        self.obs_entity_feats = 4
        self.state_entity_feats = 5

        self.longest_action_space = max(self.env_dict[TaskNames[0]].action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self.env_dict[TaskNames[0]].observation_space, key=lambda x: x.shape
        )

        # check if the scenario uses an entity state 
        self.custom_state = True
        self.custom_state_dim = self.state_entity_feats * self.max_num_entities

        self._seed = kwargs["seed"]
        for env in self.env_dict.values():
            env.seed(self._seed)

        # variables for animation
        self.t = 0

        self.agent_positions = np.zeros((self.max_num_agents, self.episode_limit, 2))
        self.landmark_positions = np.zeros((self.max_num_landmarks, self.episode_limit, 2))
        self.episode_rewards_all = np.zeros(self.episode_limit)

    def get_env(self, key, time_limit, kwargs):
        scenario = importlib.import_module("env.mpe.mpe_env.scenarios." + key).Scenario()
        task = copy.deepcopy(self.original_task)
        task.set_task(task_name=key)
        world = scenario.make_world(task=task)
        env = MultiAgentEnv(
            world, task,
            reset_callback=scenario.reset_world,
            reward_callback=scenario.reward,
            observation_callback=(
                scenario.entity_observation if kwargs.get("obs_entity_mode", False) else
                scenario.observation
            ),
            state_callback=scenario.entity_state,
            world_info_callback=getattr(scenario, "world_benchmark_data", None),
            id_callback=getattr(scenario, "get_task_id", None),
        )

        env = TimeLimit(env, max_episode_steps=time_limit)
        env = FlattenObservation(env)

        return env

    def step(self, actions, animate=False):
        """ Returns reward, terminated, info """
        actions = [int(a) for a in actions]
        self._obs, reward, done, info = self.env_dict[self.task_name].step(actions)
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]

        if self.custom_state:
            self._state = self.env_dict[self.task_name].get_state()

        # save step description if animation is required
        if animate:
            self.agent_positions[:, self.t, :] = self.get_agent_positions()
            self.landmark_positions[:, self.t, :] = self.get_landmark_positions()
            self.episode_rewards_all[self.t] = float(sum(reward))
        self.t += 1

        return float(sum(reward)), all(done), info["world"]

    def get_agent_positions(self):
        """Returns the [x,y] positions of each agent in a list"""
        return [a.state.p_pos for a in self.env_dict[self.task_name].world.agents]

    def get_landmark_positions(self):
        return [l.state.p_pos for l in self.env_dict[self.task_name].world.landmarks]

    def get_masks(self):
        agent_mask = np.zeros(self.max_num_entities)
        entity_mask = np.zeros(self.max_num_entities)

        agent_mask[:self.env_dict[self.task_name].world.num_agents] = 1
        entity_mask[:self.env_dict[self.task_name].world.num_entities] = 1

        return {'agent_mask': agent_mask, 'entity_mask': entity_mask}

    def get_task_id(self):
        task = np.zeros((self.max_num_agents, TaskIDDim))
        task[:] = self.env_dict[self.task_name].get_task_id()

        return task

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return flatdim(self.longest_observation_space)

    def get_state(self):
        if self.custom_state:
            return self._state
        else:
            return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        if self.custom_state:
            return self.custom_state_dim
        else:
            return self.max_num_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.max_num_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        valid = flatdim(self.env_dict[self.task_name].action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self):
        """ Returns initial observations and states"""
        self.task_name = TaskNames[np.random.randint(len(TaskNames))]
        self._obs = self.env_dict[self.task_name].reset()
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]
        if self.custom_state:
            self._state = self.env_dict[self.task_name].get_state()
        self.t = 0
        self.agent_positions.fill(0.)
        self.landmark_positions.fill(0.)
        self.episode_rewards_all.fill(0.)
        return self.get_obs(), self.get_state()

    def render(self):
        return self.env_dict[self.task_name].render(mode='rgb_array')

    def close(self):
        self.env_dict[self.task_name].close()

    def seed(self):
        return self.env_dict[self.task_name].seed

    def save_replay(self):
        pass

    def save_animation(self, path):
        anim = MPEAnimator(self.agent_positions[:, :self.t, :],
                           self.landmark_positions[:, :self.t, :],
                           self.episode_rewards_all[:self.t])
        anim.save_animation(path)

    def get_stats(self):
        return {}

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "obs_entity_shape": self.obs_entity_feats,
            "state_entity_shape": self.state_entity_feats,
            "n_actions": self.get_total_actions(),
            "n_agents": self.max_num_agents,
            "n_entities": self.max_num_entities,
            "episode_limit": self.episode_limit
        }
        return env_info
