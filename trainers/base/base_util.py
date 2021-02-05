import torch
from torch import nn
import os
import json


class BaseUtil:
    def __init__(self, config):
        self.c = config

    # def soft_ce(self, input, target, reduction='mean'):
    #     """
    #     :param input: (batch, *)
    #     :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
    #     """
    #     batchloss = - (target.view(target.shape[0], -1) * input.view(input.shape[0], -1)).mean()
    #     return batchloss

    def soft_ce(self, input, target, reduction='mean'):
        """
        :param input: (batch, *)
        :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
        """
        logprobs = torch.nn.functional.log_softmax(input.view(input.shape[0], -1), dim=1)
        batchloss = - torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
        if reduction == 'none':
            return batchloss
        elif reduction == 'mean':
            return torch.mean(batchloss)
        elif reduction == 'sum':
            return torch.sum(batchloss)
        else:
            raise NotImplementedError('Unsupported reduction mode.')

    def log(self, result):
        if self.rank == 0:
            for name, value in result.items():
                self.writer.add_scalar(name, float(value), self.iteration)
            del result

    def freeze_bn(self, model):
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()

    def loop(self, rank):
        raise NotImplementedError

    def save(self, result, models_dict):
        result_filename = os.path.join(self.c.weights_path,
                                       f'epoch_{self.epoch}_result.json')
        json.dump(result, open(result_filename, 'w'),
                  default=lambda o: str(o), indent=4)

        for model_name, model in models_dict.items():
            torch.save(model.state_dict(), os.path.join(
                self.c.weights_path, f'epoch_{self.epoch}_{model_name}_weights.pth'))

    def load(self, model, path):
        model.load_state_dict(torch.load(path))
