import numpy as np
from ..core import World, Agent, Landmark
from ..scenario import BaseScenario
from ..task import *


class Scenario(BaseScenario):
    def make_world(
            self,
            task=None
    ):
        world = World()
        # set any world properties first
        world.dim_c = 2
        self.num_agents = task.num_agents
        self.num_landmarks = task.num_landmarks
        self.num_entities = self.num_agents + self.num_landmarks
        self.num_entity_types = 3  # agent, landmark, self
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False

        # make initial conditions
        self.reset_world(world, task)
        self.world_benchmark = {'collisions': 0, 'min_dists': 0}

        # to access a priori the dimension of the entity state
        self.entity_obs_feats = 4
        self.entity_state_feats = 5
        self.benchmark_circles = [0.1, 0.2, 0.3, 0.4, 0.5]

        return world

    def get_task_id(self, world):
        task_id = np.zeros(TaskIDDim)

        task_id[TaskNames.index('simple_spread')] = 1
        task_id[len(TaskNames) + world.num_agents - MinNumAgentsDict['simple_spread']] = 1

        return task_id

    def reset_world(self, world, task=None):
        # random properties for agents
        world.agents = [Agent() for _ in range(task.num_agents)]
        world.landmarks = [Landmark() for _ in range(task.num_landmarks)]
        world.num_agents = task.num_agents
        world.num_landmarks = task.num_landmarks
        world.num_entities = task.num_agents + task.num_landmarks

        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False

        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = world.np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = world.np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def world_benchmark_data(self, world, final=False):
        info = self.world_benchmark.copy()
        if final:
            for c in self.benchmark_circles:
                info[f'occupied_landmarks_{c}'] = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            info['min_dists'] += min(dists)
            if final:
                for c in self.benchmark_circles:
                    if min(dists) <= c:
                        info[f'occupied_landmarks_{c}'] += 1
        if final:
            for c in self.benchmark_circles:
                info[f'occupied_landmarks_{c}'] /= len(world.landmarks)

        for agent in world.agents:
            if agent.collide:
                for a in world.agents:
                    if a is not agent and self.is_collision(a, agent):
                        info["collisions"] += 1
        return info

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        """
        A global reward equivalent to PettingZoo, modeled by rewarding only one agent
        """
        if agent.name != "agent 0":
            return 0

        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists) / len(world.landmarks)

        # for l in world.landmarks:
        #     for a in world.agents:
        #         dist = np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
        #         if dist < 0.1:
        #             rew += 1 / len(world.landmarks)

        return rew

    def entity_state(self, world):

        """
        Entity approach for transformers including velocity

        *5 features for each entity: (pos_x, pos_y, vel_x, vel_y, is_agent)*
        - agent features: (x, y, vel_x, vel_y, 1)
        - landmark features: (x, y, vel_x, vel_y, 0)
        
        In this case the absolute position of the agents is considered,
        and the velocity is also included. All these information are present
        alreay in the original state vector, because are included in the agents 
        observations which are concatenated together. 
        """

        feats = np.zeros((self.num_entities, self.entity_state_feats))

        # agents features
        i = 0
        for a in world.agents:
            pos = a.state.p_pos
            vel = a.state.p_vel
            feats[i] = [pos[0], pos[1], vel[0], vel[1], 1.]
            i += 1

        # landmarks features
        for landmark in world.landmarks:
            pos = landmark.state.p_pos
            vel = landmark.state.p_vel  # the velocity in this case is just 0
            feats[i] = [pos[0], pos[1], vel[0], vel[1], 0.]
            i += 1

        return feats

    def entity_observation(self, agent, world):

        """
        Entity approach for transformers

        *4 features for each entity: (rel_x, rel_y, is_agent, is_self)*
        - rel_x and rel_y are the relative positions of an entity in respect to agent
        - communication is not considered because is not present in this scenario
        - velocity is not included (since in the original observation an agent doesn't know other agents' velocities)

        Ex: agent 1 for agent 1:    (0, 0, 1, 1)
        Ex: agent 2 for agent 1:    (dx(a1,a2), dy(a1,a2), 1, 0)
        Ex: landmark 1 for agent 1: (dx(a1,l1), dy(a1,l1), 0, 0)
        
        """

        feats = np.zeros(self.entity_obs_feats * self.num_entities)

        # agent features
        pos_a = agent.state.p_pos
        i = 0
        for a in world.agents:
            if a is agent:
                # vel = agent.state.p_vel
                feats[i:i + self.entity_obs_feats] = [0., 0., 1., 1.]
            else:
                pos = a.state.p_pos - pos_a
                feats[i:i + self.entity_obs_feats] = [pos[0], pos[1], 1., 0.]
            i += self.entity_obs_feats

        # landmarks features
        for j, landmark in enumerate(world.landmarks):
            pos = landmark.state.p_pos - pos_a
            feats[i:i + self.entity_obs_feats] = [pos[0], pos[1], 0., 0.]
            i += self.entity_obs_feats

        return feats

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue

            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
