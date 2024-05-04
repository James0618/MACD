import numpy as np


# TaskNames = ['simple_spread', 'simple_besiege']      # ['simple_spread', 'simple_besiege']
TaskNames = ['simple_spread']      # ['simple_spread', 'simple_besiege']

MaxNumAgentsDict = {
    'simple_spread': 2,
    'simple_besiege': 9,
}
MinNumAgentsDict = {
    'simple_spread': 2,
    'simple_besiege': 3,
}
MaxNumLandmarksDict = {
    'simple_spread': 2,
    'simple_besiege': 3,
}
MinNumLandmarksDict = {
    'simple_spread': 2,
    'simple_besiege': 1,
}

TaskIDDim = 5


class Task:
    def __init__(self):
        self.name = None
        self.id = 0

        self.max_num_agents = max([MaxNumAgentsDict[key] for key in TaskNames])
        self.max_num_landmarks = max([MaxNumLandmarksDict[key] for key in TaskNames])
        self.max_num_entities = self.max_num_agents + self.max_num_landmarks

        self.num_agents = self.max_num_agents
        self.num_landmarks = self.max_num_landmarks
        self.num_targets = 0
        self.num_entities = self.max_num_entities  # num_agents + num_landmarks + num_targets
        self.type = 0  # 0,1,2

    def set_task(self, task_name):
        self.name = task_name
        self.id = np.zeros(len(TaskNames))
        self.id[TaskNames.index(self.name)] = 1

    def reset_task(self):
        max_num_agents = MaxNumAgentsDict[self.name]
        min_num_agents = MinNumAgentsDict[self.name]
        max_num_landmarks = MaxNumLandmarksDict[self.name]
        min_num_landmarks = MinNumLandmarksDict[self.name]

        if self.name == 'simple_spread':
            self.num_agents = np.random.randint(min_num_agents, max_num_agents + 1)
            self.num_landmarks = self.num_agents
            self.num_entities = self.num_agents + self.num_landmarks
        elif self.name == 'simple_besiege':
            self.num_landmarks = np.random.randint(min_num_landmarks, max_num_landmarks + 1)
            self.num_agents = 3 * self.num_landmarks
            self.num_entities = self.num_agents + self.num_landmarks
        else:
            raise NotImplementedError
