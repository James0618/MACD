import argparse
import wandb
from agent.runners.DreamerRunner import DreamerRunner
from agent.runners.SingleRunner import SingleRunner
from configs import Experiment, SimpleObservationConfig, NearRewardConfig, DeadlockPunishmentConfig, RewardsComposerConfig
from configs.EnvConfigs import StarCraftConfig, EnvCurriculumConfig, MPEConfig, MujocoConfig, transform_mujoco_name
from configs.flatland.RewardConfigs import FinishRewardConfig
from configs.dreamer.DreamerControllerConfig import DreamerControllerConfig
from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig
from configs.flatland.TimetableConfigs import AllAgentLauncherConfig
from env.flatland.params import SeveralAgents, PackOfAgents, LotsOfAgents
from environments import Env, FlatlandType, FLATLAND_OBS_SIZE, FLATLAND_ACTION_SIZE
from sentence_transformers import SentenceTransformer


def train_dreamer(exp, n_workers, mode='simple'):
    if n_workers == 1:
        runner = SingleRunner(exp.env_config, exp.learner_config, exp.controller_config, n_workers, mode)
    else:
        runner = DreamerRunner(exp.env_config, exp.learner_config, exp.controller_config, n_workers)

    if exp.learner_config.RUNNER_MODE == 'simple':
        runner.run(exp.steps, exp.episodes)
    elif exp.learner_config.RUNNER_MODE == 'dreamer':
        runner.run_ac(exp.steps, exp.episodes)
    else:
        raise NotImplementedError


def get_env_info(configs, env):
    for config in configs:
        config.IN_DIM = env.n_obs
        config.ACTION_SIZE = env.n_actions
    env.close()


def get_env_info_flatland(configs):
    for config in configs:
        config.IN_DIM = FLATLAND_OBS_SIZE
        config.ACTION_SIZE = FLATLAND_ACTION_SIZE


def prepare_mpe_configs(mode):
    agent_configs = [DreamerControllerConfig(), DreamerLearnerConfig()]
    env_config = MPEConfig(mode)
    get_env_info(agent_configs, env_config.create_env())
    for config in agent_configs:
        config.IF_DISCRETE = True
    return {"env_config": (env_config, 100),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}


def prepare_starcraft_configs(env_name):
    agent_configs = [DreamerControllerConfig(), DreamerLearnerConfig()]
    env_config = StarCraftConfig(env_name)
    get_env_info(agent_configs, env_config.create_env())
    for config in agent_configs:
        config.IF_DISCRETE = True
    return {"env_config": (env_config, 100),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}


def prepare_mujoco_configs(env_name, seed=0, use_modified_state=False, device='cuda:0', with_multi_task=False):
    agent_configs = [DreamerControllerConfig(), DreamerLearnerConfig()]
    mujoco_configs = transform_mujoco_name(env_name)
    if with_multi_task:
        sbert_model = SentenceTransformer('sbert/all-MiniLM-L6-v2').to(device)
    else:
        sbert_model = None
    env_config = MujocoConfig(
        scenario=mujoco_configs['scenario'], agent_conf=mujoco_configs['agent_conf'],
        agent_obsk=mujoco_configs['agent_obsk'], seed=seed, sbert_model=sbert_model,
        use_modified_state=use_modified_state
    )
    # get_env_info(agent_configs, env_config.create_env())
    temp_env = env_config.create_env()
    for config in agent_configs:
        config.IN_DIM = temp_env.observation_space[0].shape[0]
        if sbert_model is not None:
            config.TASK_DIM = sbert_model.get_sentence_embedding_dimension()
        else:
            config.TASK_DIM = 0
        config.CENTRAL_OBS_SIZE = temp_env.share_observation_space[0].shape[0]
        config.ACTION_SIZE = temp_env.action_space[0].shape[0]
        config.IF_DISCRETE = False
    temp_env.close()

    return {"env_config": (env_config, 100),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}


def prepare_flatland_configs(env_name):
    if env_name == FlatlandType.FIVE_AGENTS:
        env_config = SeveralAgents(RANDOM_SEED + 100)
    elif env_name == FlatlandType.TEN_AGENTS:
        env_config = PackOfAgents(RANDOM_SEED + 100)
    elif env_name == FlatlandType.FIFTEEN_AGENTS:
        env_config = LotsOfAgents(RANDOM_SEED + 100)
    else:
        raise Exception("Unknown flatland environment")
    obs_builder_config = SimpleObservationConfig(max_depth=3, neighbours_depth=3,
                                                 timetable_config=AllAgentLauncherConfig())
    reward_config = RewardsComposerConfig((FinishRewardConfig(finish_value=10),
                                           NearRewardConfig(coeff=0.01),
                                           DeadlockPunishmentConfig(value=-5)))
    agent_configs = [DreamerControllerConfig(), DreamerLearnerConfig()]
    get_env_info_flatland(agent_configs)
    return {"env_config": (env_config, 100),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": reward_config,
            "obs_builder_config": obs_builder_config}


def add_attributes(config, name, value):
    config.__setattr__(name, value)


if __name__ == "__main__":
    RANDOM_SEED = 23
    parser = argparse.ArgumentParser()
    # flatland, starcraft, mpe
    parser.add_argument('--env', type=str, default="mujoco", help='mpe, starcraft, or mujoco')
    parser.add_argument('--env_name', type=str, default="2-agent_halfcheetah", help='Specific setting')
    parser.add_argument('--mode', type=str, default='simple', help='test, simple, on-policy, and off-policy')
    parser.add_argument('--device', type=str, default='cuda:0', help='from cuda:0 to cuda:3')
    parser.add_argument('--exp_id', type=str, default='nothing', help='the name of the experiment')
    parser.add_argument('--project_id', type=str, default='nothing', help='the name of the experiment')
    parser.add_argument(
        '--agent_mode', type=str, default='simple', help='simple, shapley-critic, shapley-rollout, or dae'
    )
    parser.add_argument('--notes', type=str, default='nothing', help='the notes of the experiment')
    parser.add_argument('--runner_mode', type=str, default='simple',
                        help='simple or dreamer, the runner with dreamer mode update the policy every n steps')
    parser.add_argument('--ac_mode', type=str, default='simple', help='simple or dreamer')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--train_ratio', type=int, default=100, help='The ratio of training times to sampling steps')
    parser.add_argument('--use_consistency', action='store_true', help='If use the shapley value to update the actor')
    parser.add_argument('--use_modified_state', action='store_true', help='If use the modified central state')
    parser.add_argument('--with_multi_task', action='store_true', help='If use the multi-task setting')
    parser.add_argument(
        '--reconstruct_central', action='store_true', help='If use the central state as the reconstruction target'
    )
    args = parser.parse_args()

    if args.env == Env.FLATLAND:
        configs = prepare_flatland_configs(args.env_name)
    elif args.env == Env.STARCRAFT:
        configs = prepare_starcraft_configs(args.env_name)
    elif args.env == Env.MPE:
        if args.mode == 'on-policy':
            env_mode = 'entity'
        else:
            env_mode = 'simple'
        configs = prepare_mpe_configs(env_mode)
    elif args.env == Env.MUJOCO:
        configs = prepare_mujoco_configs(
            args.env_name, seed=RANDOM_SEED, use_modified_state=args.use_modified_state, device=args.device,
            with_multi_task=args.with_multi_task
        )
    else:
        raise Exception("Unknown environment")

    keys = ['env_config', 'controller_config', 'learner_config']

    configs["env_config"][0].ENV_TYPE = Env(args.env)

    for key in keys[1:]:
        add_attributes(configs[key], 'ENV_TYPE', Env(args.env))
        add_attributes(configs[key], 'DEVICE', args.device)
        add_attributes(configs[key], 'AGENT_MODE', args.agent_mode)
        add_attributes(configs[key], 'USE_CONSISTENCY', args.use_consistency)
        add_attributes(configs[key], 'NOTES', args.notes)
        add_attributes(configs[key], 'RECONSTRUCT_CENTRAL', args.reconstruct_central)
        add_attributes(configs[key], 'RUNNER_MODE', args.runner_mode)
        add_attributes(configs[key], 'AC_MODE', args.ac_mode)
        add_attributes(configs[key], 'TRAIN_RATIO', args.train_ratio)
        add_attributes(configs[key], 'MULTI_TASK', args.with_multi_task)
        if args.exp_id != 'nothing':
            add_attributes(configs[key], 'EXP_ID', args.exp_id)
        if args.project_id != 'nothing':
            add_attributes(configs[key], 'PROJECT_ID', args.project_id)

    wandb.login(key='de4ca62cf3d2dbb5773336a1d4ad4ef47a8692c1')

    exp = Experiment(steps=10 ** 10,
                     episodes=50000,
                     random_seed=RANDOM_SEED,
                     env_config=EnvCurriculumConfig(*zip(configs["env_config"]), Env(args.env),
                                                    obs_builder_config=configs["obs_builder_config"],
                                                    reward_config=configs["reward_config"]),
                     controller_config=configs["controller_config"],
                     learner_config=configs["learner_config"])

    train_dreamer(exp, n_workers=args.n_workers, mode=args.mode)
