from functools import partial

from .mpe_env.mpe_wrapper import MPEWrapper
from .multiagentenv import MultiAgentEnv


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {
    "mpe_env": partial(env_fn, env=MPEWrapper),
}

