from trainers.original_trainer import OriginalTrainer
from dotmap import DotMap
import datasets
import config_datasets
import models
import config_models
from torch.optim import AdamW

config = DotMap()

config.trainer = OriginalTrainer
config.epochs = 25
config.log_train_every = 100
config.master_addr = '127.0.0.1'
config.master_port = '29800'
config.seed = 17
config.amp = True
config.data_workers = 8

config.opt = AdamW
config.opt_params.max_lr = 2 * 1e-4
config.opt_params.weight_decay = 1e-2

config.w_chamfer = 0.1

config.dataset_cls = datasets.KITTI
config.train_dataset_params = config_datasets.kitti_train_params
config.test_dataset_params = config_datasets.kitti_test_params

config.model = models.UnetAdaptiveBins.build
config.model_params = config_models.adabin_params

config.min_depth_eval = 1e-3
config.max_depth_eval = 80