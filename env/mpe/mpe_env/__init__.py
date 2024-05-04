import importlib
from . import core, scenario
from gym.envs.registration import register
from . import scenarios
from .task import Task


# Multiagent envs
# ----------------------------------------

# _particles = {
#     "simple_adversary": "SimpleAdversary-v0",
#     "simple_crypto": "SimpleCrypto-v0",
#     "simple_push": "SimplePush-v0",
#     "simple_spread": "SimpleSpread-v0",
#     "simple_tag": "SimpleTag-v0",
#     "climbing_spread": "ClimbingSpread-v0",
# }
_particles = {
    "simple_spread": "SimpleSpread-v0",
    "simple_besiege": "SimpleBesiege-v0",
}
original_task = Task()

for scenario_name, gymkey in _particles.items():
    scenario_module = importlib.import_module("env.mpe.mpe_env.scenarios."+scenario_name)
    scenario_aux = scenario_module.Scenario()
    world = scenario_aux.make_world(task=original_task)

    # Registers multi-agent particle environments:
    register(
        gymkey,
        entry_point="env.mpe.mpe_env.environment:MultiAgentEnv",
        kwargs={
            "world": world,
            "reset_callback": scenario_aux.reset_world,
            "reward_callback": scenario_aux.reward,
            "observation_callback": scenario_aux.observation,
            "world_info_callback": getattr(scenario_aux, "world_benchmark_data", None)
        },
    )
