from agent.controllers.DreamerController import DreamerController
from agent.controllers.ContinuousController import ContinuousController
from agent.controllers.DreamerV3Controller import DreamerController as DreamerV3Controller
from agent.controllers.NewController import DreamerController as NewController
from configs.dreamer.DreamerAgentConfig import DreamerConfig


class DreamerControllerConfig(DreamerConfig):
    def __init__(self):
        super().__init__()

        self.EXPL_DECAY = 0.998
        self.EXPL_NOISE = 0.9
        self.EXPL_MIN = 0.05
        self.MAX_NUM_ENTITIES = 2

    def create_controller(self, mode='simple', max_num_entities=2):
        self.MAX_NUM_ENTITIES = max_num_entities
        if mode == 'on-policy':
            return NewController(self)
        elif mode == 'simple':
            if self.continuous:
                return ContinuousController(self)
            else:
                return DreamerController(self)
        elif mode == 'off-policy':
            return DreamerV3Controller(self)
        elif mode == 'test':
            raise NotImplementedError
