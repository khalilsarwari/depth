from trainers.original_trainer import OriginalTrainer
from dotmap import DotMap
import datasets
import config_datasets
import models
import config_models
from torch.optim import AdamW

config = DotMap()

config.trainer = OriginalTrainer
config.epochs = 5
config.log_train_every = 100
config.master_addr = '127.0.0.1'
config.master_port = '29800'
config.seed = 17
config.amp = True
config.data_workers = 16
config.notes = 'data vs performance for TOD'
config.commit = '7982dd5'

config.opt = AdamW
config.opt_params.max_lr = 2 * 1e-4
config.opt_params.weight_decay = 1e-2

config.w_chamfer = 0

# config.dataset_cls = datasets.KITTI
# config.train_dataset_params = config_datasets.kitti_train_params
# config.test_dataset_params = config_datasets.kitti_test_params

# config.dataset_cls = datasets.NYU
# config.train_dataset_params = config_datasets.nyu_train_params
# config.test_dataset_params = config_datasets.nyu_test_params

config.dataset_cls = datasets.TODD
config.train_dataset_params = config_datasets.todd_train_params
config.test_dataset_params = config_datasets.todd_test_params

config.model = models.Custom.build
config.model_params = config_models.custom_params

config.min_depth_eval = 1e-3
config.max_depth_eval = 260

# config.min_depth_eval = 1e-3
# config.max_depth_eval = 10