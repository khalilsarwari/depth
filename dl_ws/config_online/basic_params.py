from dotmap import DotMap
import models
import config_models
import datasets
import config_datasets
config = DotMap()

config.dataset_cls = datasets.NYU
config.train_dataset_params = config_datasets.nyu_train_params
config.test_dataset_params = config_datasets.nyu_test_params

config.model = models.Custom.build
config.model_params = config_models.custom_params