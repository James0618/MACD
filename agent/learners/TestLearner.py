from copy import deepcopy
from pathlib import Path
import torch
from agent.memory.EntityMemory import DreamerMemory
from agent.models.TransfDreamer import DreamerModel
from networks.dreamer.action import EntityActor
from networks.dreamer.critic import EntityPPOCritic


def orthogonal_init(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


def initialize_weights(mod, scale=1.0, mode='ortho'):
    for p in mod.parameters():
        if mode == 'ortho':
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
        elif mode == 'xavier':
            if len(p.data.shape) >= 2:
                torch.nn.init.xavier_uniform_(p.data)


class DreamerLearner:
    def __init__(self, config):
        self.config = config
        self.model = DreamerModel(config).to(config.DEVICE).eval()
        self.actor = EntityActor(
            config.FEAT, config.ACTION_SIZE, config.ACTION_HIDDEN, config.MAX_NUM_ENTITIES
        ).to(config.DEVICE)
        # self.critic = MADDPGCritic(config.FEAT, config.HIDDEN).to(config.DEVICE)
        self.critic = EntityPPOCritic(config.FEAT, config.HIDDEN).to(config.DEVICE)
        initialize_weights(self.model, mode='xavier')
        initialize_weights(self.actor)
        initialize_weights(self.critic, mode='xavier')
        self.old_critic = deepcopy(self.critic)
        self.replay_buffer = DreamerMemory(config.CAPACITY, config.SEQ_LENGTH, config.ACTION_SIZE, config.IN_DIM, 2,
                                           config.DEVICE, config.ENV_TYPE)
        self.entropy = config.ENTROPY
        self.step_count = -1
        self.cur_update = 1
        self.accum_samples = 0
        self.total_samples = 0
        self.init_optimizers()
        self.n_agents = 2
        Path(config.LOG_FOLDER).mkdir(parents=True, exist_ok=True)

    def init_optimizers(self):
        pass

    def params(self):
        return {'model': {k: v.cpu() for k, v in self.model.state_dict().items()},
                'actor': {k: v.cpu() for k, v in self.actor.state_dict().items()},
                'critic': {k: v.cpu() for k, v in self.critic.state_dict().items()}}

    def step(self, rollout, if_train=False):
        pass

    def train_model(self, samples, if_log=False):
        pass

    def train_agent(self, samples, if_log=False):
        pass

    def apply_optimizer(self, opt, model, loss, grad_clip):
        pass
