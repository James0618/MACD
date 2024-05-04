from agent.learners.DreamerLearner import DreamerLearner
from agent.learners.ContinuousLearner import ContinuousLearner
from agent.learners.EntityLearner import DreamerLearner as EntityLearner
from agent.learners.OffPolicyLearner import DreamerLearner as OffPolicyLearner
from agent.learners.TestLearner import DreamerLearner as TestLearner
from configs.dreamer.DreamerAgentConfig import DreamerConfig


class DreamerLearnerConfig(DreamerConfig):
    def __init__(self):
        super().__init__()
        self.EXP_ID = 'debug'
        self.PROJECT_ID = 'debug'
        self.MODEL_LR = 4e-4
        self.Q_LR = 5e-4
        self.ACTOR_LR = 5e-4
        self.VALUE_LR = 5e-4
        self.CAPACITY = 250000
        self.MIN_BUFFER_SIZE = 1000
        self.MODEL_EPOCHS = 100         # 60
        self.EPOCHS = 5                 # 4
        self.PPO_EPOCHS = 5             # 5
        self.MODEL_BATCH_SIZE = 40      # 40
        self.BATCH_SIZE = 16            # 40
        self.SEQ_LENGTH = 40
        self.N_SAMPLES = 1
        self.TARGET_UPDATE = 1
        self.Q_TARGET_UPDATE = 10
        self.DEVICE = 'cuda:0'
        self.GRAD_CLIP = 10.0
        self.PPO_CLIP_COEF = 0.2
        self.HORIZON = 15
        self.SHAPLEY_HORIZON = 8
        self.ENTROPY = 0.01
        self.ENTROPY_ANNEALING = 0.9995
        self.MAX_SHAPLEY_SAMPLE_NUM = 1
        self.GRAD_CLIP_POLICY = 10.0
        self.TIME_LIMIT = 25
        self.TAU = 0.995
        self.ALPHA = 0.05
        self.slow_target_fraction = 0.02
        self.layer_mode = 'dreamer-v3'

    def create_learner(self, mode='simple', n_agents=1):
        # mode: test, simple, on-policy, and off-policy
        if mode == 'test':
            return TestLearner(self)
        elif mode == 'simple':
            if self.continuous:
                return ContinuousLearner(self, n_agents=n_agents)
            else:
                return DreamerLearner(self, n_agents=n_agents)
        elif mode == 'on-policy':
            return EntityLearner(self)
        elif mode == 'off-policy':
            return OffPolicyLearner(self)
        else:
            raise NotImplementedError
