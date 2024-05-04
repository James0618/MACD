from agent.workers.TestWorker import SingleWorker
from agent.workers.EntityWorker import SingleWorker as EntityWorker
import matplotlib.animation as animation
import matplotlib.pyplot as plt


class SingleServer:
    def __init__(self, env_config, controller_config, new=False):
        if new:
            self.workers = EntityWorker(0, env_config, controller_config, new=True)
        else:
            self.workers = SingleWorker(0, env_config, controller_config)

    def run(self, model):
        recvs = self.workers.run(model)
        return recvs


class Runner:
    def __init__(self, env_config, learner_config, controller_config, n_workers, if_new_model=False):
        self.n_workers = n_workers
        self.server = SingleServer(env_config, controller_config, new=if_new_model)
        if if_new_model:
            learner_config.MAX_NUM_ENTITIES = self.server.workers.max_num_entities
        else:
            learner_config.MAX_NUM_ENTITIES = 2

        self.learner = learner_config.create_learner(new=if_new_model, test=True)

    def run(self, max_steps=10 ** 10, max_episodes=10 ** 10):
        cur_steps, cur_episode = 0, 0
        episode_reward, episode_return = 0, 0

        while True:
            fig, ax = plt.subplots()
            rollout, info = self.server.run(self.learner.params())
            episode_reward += info["reward"]
            episode_return += info["return"]

            cur_steps += info["steps_done"]
            cur_episode += 1
            print("Episode: {}, Timestep: {}, Reward: {}, Return: {}".format(
                cur_episode - 1, self.learner.total_samples, episode_reward, episode_return)
            )
            episode_reward, episode_return = 0, 0

            images = []
            for image in info['images']:
                images.append([ax.imshow(image, animated=True)])

            ani = animation.ArtistAnimation(fig, images, interval=50)
            writer = animation.ImageMagickFileWriter()
            ani.save('demos/demo-{}.gif'.format(cur_episode), writer=writer)

            if cur_episode >= max_episodes or cur_steps >= max_steps:
                break

