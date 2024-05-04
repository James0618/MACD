import wandb
import torch.nn as nn
from agent.workers.SingleWorker import SingleWorker
from agent.workers.ContinuousWorker import ContinuousWorker
from agent.workers.EntityWorker import SingleWorker as EntityWorker


class SingleServer:
    def __init__(self, env_config, controller_config, mode='simple'):
        if mode == 'on-policy':
            self.workers = EntityWorker(0, env_config, controller_config, mode=mode)
        else:
            self.workers = SingleWorker(0, env_config, controller_config, mode=mode)

    def run(self, model):
        recvs = self.workers.run(model)
        return recvs

    def evaluate(self, model):
        # not be used! only for compatibility
        recvs = self.workers.run(model)
        return recvs


class ContinuousServer:
    def __init__(self, env_config, controller_config, mode='simple'):
        self.workers = ContinuousWorker(0, env_config, controller_config, mode=mode)

    def run(self, model):
        recvs = self.workers.run(model)
        return recvs

    def run_ac(self, model, start, last_state, last_central_state, episode_return):
        recvs = self.workers.run_ac(
            model, start=start, last_state=last_state, last_central_state=last_central_state,
            episode_return=episode_return
        )
        return recvs

    def evaluate(self, model):
        recvs = self.workers.evaluate(model)
        return recvs


class SingleRunner:
    def __init__(self, env_config, learner_config, controller_config, n_workers, mode='simple'):
        self.n_workers = n_workers
        if learner_config.layer_mode == 'dreamer-v3':
            learner_config.activation, learner_config.norm = nn.SiLU, nn.LayerNorm
            controller_config.activation, controller_config.norm = nn.SiLU, nn.LayerNorm
        else:
            learner_config.activation, learner_config.norm = nn.ReLU, nn.Identity
            controller_config.activation, controller_config.norm = nn.ReLU, nn.Identity

        if env_config.ENV_TYPE.value == 'mujoco':
            learner_config.continuous, controller_config.continuous = True, True
            self.server = ContinuousServer(env_config, controller_config, mode=mode)
        else:
            learner_config.continuous, controller_config.continuous = False, False
            self.server = SingleServer(env_config, controller_config, mode=mode)

        if mode == 'on-policy':
            learner_config.MAX_NUM_ENTITIES = self.server.workers.max_num_entities
        else:
            learner_config.MAX_NUM_ENTITIES = 2

        self.learner = learner_config.create_learner(mode=mode, n_agents=self.server.workers.n_agents)

    def run(self, max_steps=10 ** 10, max_episodes=10 ** 10):
        cur_steps, cur_episode = 0, 0
        while True:
            rollout, info = self.server.run(self.learner.params())
            # evaluate_info = self.server.evaluate(self.learner.params())
            if self.learner.config.AC_MODE == 'simple':
                self.learner.step(rollout)
            elif self.learner.config.AC_MODE == 'dreamer':
                self.learner.step_ac(rollout)
            else:
                raise NotImplementedError
            cur_steps += info["steps_done"]
            cur_episode += 1
            evaluate_info = {'return': 0}
            wandb.log({
                'return': info["return"], 'evaluate return': evaluate_info['return'],
                'timestep': self.learner.total_samples
            })
            print("Episode: {}, Timestep: {}, Return: {}, Evaluate return: {}".format(
                cur_episode - 1, self.learner.total_samples, info["return"], evaluate_info['return'])
            )

            if cur_episode >= max_episodes or cur_steps >= max_steps:
                break

    def run_ac(self, max_steps=10 ** 10, max_episodes=10 ** 10):
        cur_steps, cur_episode = 0, 0
        while True:
            done_flag, start = False, True
            state, central_state, episode_return, steps_done = None, None, 0, 0
            while not done_flag:
                rollout, info = self.server.run_ac(
                    self.learner.params(), start, state, central_state, episode_return
                )
                self.learner.step_ac(rollout)
                start = False
                done_flag, state, central_state = info['done_flag'], info['state'], info['central_state']
                episode_return, steps_done = info['return'], info['steps_done']

            evaluate_info = self.server.evaluate(self.learner.params())
            cur_steps += info["steps_done"]
            cur_episode += 1
            wandb.log({
                'return': info["return"], 'evaluate return': evaluate_info['return'],
                'timestep': self.learner.total_samples
            })
            print("Episode: {}, Timestep: {}, Return: {}, Evaluate return: {}".format(
                cur_episode - 1, self.learner.total_samples, info["return"], evaluate_info['return'])
            )

            if cur_episode >= max_episodes or cur_steps >= max_steps:
                break