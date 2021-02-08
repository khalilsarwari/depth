import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
import torchvision

from tqdm import tqdm
import os
from collections import defaultdict
import numpy as np

from .base_trainer import BaseTrainer
import torch.distributed as dist
from utils import set_seed
from torch.optim.lr_scheduler import OneCycleLR
import helpers
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import losses

# print('comment below to not leak memory!')
# torch.autograd.set_detect_anomaly(True)

class OriginalTrainer(BaseTrainer):
    """ Original Trainer """

    def __init__(self, config):
        super().__init__(config)
        self.criterion_ueff = losses.SILogLoss()
        self.criterion_bins = losses.BinsChamferLoss()

    def init_dataloaders(self):

        self.train_dataset = self.c.dataset_cls(self.c.train_dataset_params)
        sampler = DistributedSampler(self.train_dataset) if torch.cuda.device_count() > 1 else None
        self.train_dataloader = DataLoader(self.train_dataset, 
                                    batch_size=self.c.train_dataset_params.batch_size,
                                    sampler=sampler,
                                    num_workers=self.c.data_workers, 
                                    drop_last=False, 
                                    pin_memory=True)

        self.test_dataset = self.c.dataset_cls(self.c.test_dataset_params)
        sampler = DistributedSampler(self.test_dataset) if torch.cuda.device_count() > 1 else None
        self.test_dataloader = DataLoader(self.test_dataset, 
                                    batch_size=self.c.test_dataset_params.batch_size,
                                    sampler=sampler, 
                                    num_workers=self.c.data_workers, 
                                    drop_last=False, 
                                    pin_memory=True)

    def train(self, train_batch):
        self.model.train()

        train_results = {}

        losses = {}
        stats = {}
        vis = train_batch
        with autocast(enabled=self.c.amp):
            y = train_batch['y_aug'].cuda()
            x = train_batch['x_aug'].cuda()

            bin_edges, pred = self.model(x)
            vis['pred_y'] = nn.functional.interpolate(pred, y.shape[-2:], mode='bilinear', align_corners=True)

            mask = y > self.c.model_params.min_val
            losses['l_dense'] = self.criterion_ueff(pred, y, mask=mask.to(torch.bool), interpolate=True)
            losses['l_chamfer'] = self.criterion_bins(bin_edges, y)

        losses['total_loss'] = losses['l_dense'] + self.c.w_chamfer * losses['l_chamfer']

        outputs = {}
        outputs['losses'] = losses
        outputs['vis'] = vis
        outputs['stats'] = stats

        self.scaler.scale(outputs['losses']['total_loss']).backward()

        if self.iteration % self.c.log_train_every == 0:
            self.visualize_batch_result(outputs['vis'], prefix="train_batch")

        for k, v in outputs['losses'].items():
            train_results['train/'+k] = v
        for k, v in outputs['stats'].items():
            train_results['train/'+k] = v

        train_results['lr'] = self.scheduler.get_last_lr()[0]
        return train_results

    def test(self):
        self.model.eval()
        test_losses = {}
        with torch.no_grad():
            test_losses = self.test_on_loader(self.test_dataloader)
            test_losses.update(test_losses)

        if self.rank == 0:
            self.save(test_losses, {'model': self.model})
        return test_losses

    def test_on_loader(self,test_loader):
        with torch.no_grad():
            val_si = helpers.RunningAverage()
            metrics = helpers.RunningAverageDict()
            for batch in test_loader:
                img = batch['x'].cuda()
                depth = batch['y'].cuda()
                assert depth.shape[0] == 1, 'use batch size 1 for testing'
                depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
                bins, pred = self.model(img)

                mask = depth > self.c.model_params.min_val
                l_dense = self.criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=True)
                val_si.append(l_dense)

                pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)

                pred = pred.squeeze()
                pred[pred < self.c.min_depth_eval] =  self.c.min_depth_eval
                pred[pred > self.c.max_depth_eval] = self.c.max_depth_eval
                pred[torch.isinf(pred)] =self.c.max_depth_eval
                pred[torch.isnan(pred)] = self.c.min_depth_eval

                gt_depth = depth.squeeze()
                valid_mask = torch.logical_and(gt_depth > 1e-3, gt_depth < 80)

                # garg_crop:
                gt_height, gt_width = gt_depth.shape
                eval_mask = torch.zeros(valid_mask.shape).cuda()
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                valid_mask = torch.logical_and(valid_mask, eval_mask)
                metrics.update(helpers.compute_errors(gt_depth[valid_mask], pred[valid_mask]))
                if self.rank == 0:
                    self.pbar.set_description("Epoch {}/{} | Loss {}".format(self.epoch, self.c.epochs, val_si.get_value().item()))
            test_losses = metrics.get_value()
            test_losses['val_si'] = val_si.get_value()

            # post-process losses
            for k, v in test_losses.items():
                dist.all_reduce(test_losses[k])
                test_losses[k] = test_losses[k]/torch.cuda.device_count()

            return test_losses


    def get_next_train_batches(self):
        try:
            train_batch = next(self.train_dataloader_iter)
        except StopIteration:
            self.train_dataloader_iter = iter(
                self.train_dataloader)
            train_batch = next(self.train_dataloader_iter)

        return train_batch

    def loop(self, rank):
        self.rank = rank
        if self.rank == 0:
            self.writer = SummaryWriter(self.c.exp_path)
        else:
            self.writer = None

        self.init_dataloaders()
        self.init_model()
        self.total_iterations = self.c.epochs * \
            len(self.train_dataloader)
        if torch.cuda.device_count() > 1:
            mdl = self.model.module
        else:
            mdl = self.model
        self.opt = self.c.opt([{"params": mdl.get_1x_lr_params(), "lr": self.c.opt_params.max_lr / 10},
                  {"params": mdl.get_10x_lr_params(), "lr": self.c.opt_params.max_lr}], weight_decay = self.c.opt_params.weight_decay)
        self.scheduler = OneCycleLR(self.opt, max_lr=self.c.opt_params.max_lr, 
                                                total_steps=self.total_iterations,
                                                cycle_momentum=True,
                                                base_momentum=0.85, max_momentum=0.95, 
                                                last_epoch=-1,
                                                final_div_factor=100
                                    )
        self.pbar = tqdm(total=self.total_iterations)
        self.epoch = 0
        self.train_dataloader_iter = iter(self.train_dataloader)
        while self.iteration <= self.total_iterations:
            train_batch = self.get_next_train_batches()

            if self.iteration % len(self.train_dataloader) == 0:
                test_losses = self.test()
                self.log(test_losses)
                self.epoch += 1

            result = self.train(train_batch)

            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            self.scheduler.step()

            if self.iteration % self.c.log_train_every == 0:
                self.log(result)
            self.iteration += 1
            if self.rank == 0:
                self.pbar.set_description("Epoch {}/{}, {:.1%} | Loss {}".format(
                    self.epoch, 
                    self.c.epochs, 
                    (self.iteration % len(self.train_dataloader))/len(self.train_dataloader),
                    result['train/total_loss'].item()))
                self.pbar.update(1)

            del result

        self.test()
