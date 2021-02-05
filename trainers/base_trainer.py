from .base import BaseModel, BaseTest, BaseUtil, BaseVis
from utils import *
import torch.multiprocessing as mp


class BaseTrainer(BaseModel, BaseTest, BaseUtil, BaseVis):
    def __init__(self, config):
        self.c = config
        self.iteration = 0

    def loop(self, rank):
        raise NotImplementedError

    def run(self):
        # launch processes
        mp.spawn(init_processes, args=(self.loop, self.c),
                 nprocs=torch.cuda.device_count(), join=True,
                 daemon=False)
