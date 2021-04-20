import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
from torch.cuda.amp import GradScaler
from torch.optim import Adam, AdamW, SGD
from torch.nn.parallel import DistributedDataParallel

class BaseModel:
    def __init__(self, config):
        self.c = config
        self.iteration = 0

    def init_model(self):
        self.model = self.c.model(**self.c.model_params).cuda()

        # pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print('Total trainable params:', pytorch_total_params)
            
        if torch.cuda.device_count() > 1:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DistributedDataParallel(self.model, device_ids=[self.rank], find_unused_parameters=True)

        if self.c.checkpoint:
            print('Loading from checkpoint:', self.c.checkpoint)
            state_dict = torch.load(self.c.checkpoint, map_location=f"cuda:{self.rank}")
            self.model.load_state_dict(state_dict)

        self.scaler = GradScaler(enabled=self.c.amp)