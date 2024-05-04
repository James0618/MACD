from functools import partial
from .mpe_env.mpe_wrapper import MPEWrapper
from .multiagentenv import MultiAgentEnv


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


class MPE:
    def __init__(self, obs_mode='simple'):
        self.obs_mode = obs_mode
        if obs_mode == 'simple':
            self.env = partial(env_fn, env=MPEWrapper)(
                key='simple_spread', map_name='mpe', time_limit=25, seed=0,
                obs_entity_mode=False
            )
        elif obs_mode == 'entity':
            self.env = partial(env_fn, env=MPEWrapper)(
                key='simple_spread', map_name='mpe', time_limit=25, seed=0,
                obs_entity_mode=True
            )
        else:
            raise ValueError('Invalid obs_mode: {}'.format(obs_mode))

        env_info = self.env.get_env_info()
        if obs_mode == 'simple':
            self.n_obs = env_info["obs_shape"]
        elif obs_mode == 'entity':
            self.n_obs = env_info["state_entity_shape"]
        else:
            raise ValueError('Invalid obs_mode: {}'.format(obs_mode))

        self.n_actions = env_info["n_actions"]
        self.n_agents = env_info["n_agents"]
        self.n_entities = env_info["n_entities"]
        self.episode_limit = env_info["episode_limit"]

    def to_dict(self, l):
        return {i: e for i, e in enumerate(l)}

    def step(self, action_dict):
        reward, done, info = self.env.step(action_dict)
        if self.obs_mode == 'simple':
            return self.to_dict(self.env.get_obs()), {i: reward for i in range(self.n_agents)}, \
                {i: done for i in range(self.n_agents)}, info
        else:
            return self.env.get_state(), reward, done, info

    def reset(self):
        self.env.reset()
        if self.obs_mode == 'simple':
            return {i: obs for i, obs in enumerate(self.env.get_obs())}
        else:
            return self.env.get_state()

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def get_avail_agent_actions(self, handle):
        return self.env.get_avail_agent_actions(handle)

    def get_masks(self):
        return self.env.get_masks()
