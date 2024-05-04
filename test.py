import torch
import argparse
from agent.runners.TestRunner import Runner
from configs import Experiment
from configs.EnvConfigs import EnvCurriculumConfig
from environments import Env
from train import prepare_mpe_configs, prepare_starcraft_configs, prepare_flatland_configs
from pyvirtualdisplay import Display


_display = Display(visible=0, size=(1400, 900))
_display.start()


def test(load_path, if_new_model=False):
    runner = Runner(exp.env_config, exp.learner_config, exp.controller_config, 1, if_new_model)
    params = torch.load('wandb/wandb/run-{}/files/model.pt'.format(load_path))
    runner.learner.model.load_state_dict(params['model'])
    runner.learner.actor.load_state_dict(params['actor'])
    runner.learner.critic.load_state_dict(params['critic'])

    runner.run(exp.steps, exp.episodes)


if __name__ == '__main__':
    RANDOM_SEED = 23
    parser = argparse.ArgumentParser()
    # flatland, starcraft, mpe
    parser.add_argument('--env', type=str, default="mpe", help='Flatland or SMAC env')
    parser.add_argument('--env_name', type=str, default="simple_spread", help='Specific setting')
    parser.add_argument('--mode', type=str, default='entity', help='simple or entity')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers')
    args = parser.parse_args()

    if args.mode == 'simple':
        if_new_model = False
    else:
        if_new_model = True

    if args.env == Env.FLATLAND:
        configs = prepare_flatland_configs(args.env_name)
    elif args.env == Env.STARCRAFT:
        configs = prepare_starcraft_configs(args.env_name)
    elif args.env == Env.MPE:
        configs = prepare_mpe_configs(args.mode)
    else:
        raise Exception("Unknown environment")

    configs["env_config"][0].ENV_TYPE = Env(args.env)
    configs["learner_config"].ENV_TYPE = Env(args.env)
    configs["controller_config"].ENV_TYPE = Env(args.env)

    exp = Experiment(steps=10 ** 10,
                     episodes=5,
                     random_seed=RANDOM_SEED,
                     env_config=EnvCurriculumConfig(*zip(configs["env_config"]), Env(args.env),
                                                    obs_builder_config=configs["obs_builder_config"],
                                                    reward_config=configs["reward_config"]),
                     controller_config=configs["controller_config"],
                     learner_config=configs["learner_config"])

    test(load_path='20230613_143415-bqp2j1wi', if_new_model=if_new_model)
