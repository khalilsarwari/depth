import torch
from collections import defaultdict
from torch.cuda.amp import autocast


class BaseTest:
    def __init__(self, config):
        self.c = config

    def test_on_loader(self, loader, prefix):
        test_losses = defaultdict(lambda: 0)
        test_ious = defaultdict(lambda: 0)
        for i, batch in enumerate(loader, start=1):
            losses = {}
            stats = {}
            vis = batch
            with torch.no_grad():
                with autocast(enabled=self.c.amp):
                    logits = self.model(batch['x'].cuda())
                    loss = self.ce(logits, batch['y'].cuda())
                    losses['seg_loss'] = loss
                    vis['logits'] = logits
                    prob, ind = self.softmax(logits).max(1)
                vis['prob'] = prob
                vis['ind'] = ind
            losses['total_loss'] = sum(v for v in losses.values())

            outputs = {}
            outputs['losses'] = losses
            outputs['vis'] = vis
            outputs['stats'] = stats
            self.visualize_batch_result(
                outputs['vis'], prefix=f"{prefix}_batch_{i}")

            # process losses
            losses = outputs['losses']
            for k, v in losses.items():
                test_losses[f'{prefix}_{k}'] += v

            # process ious
            ious = self.process_ious_batch(outputs['vis'])
            for k, v in ious.items():
                test_ious[k] += v

            if self.rank == 0:
                self.pbar.set_postfix({'test_progress': i/len(loader), 'len': len(loader)})
                self.pbar.set_description(
                    f"Epoch {self.epoch}/{self.c.epochs} | {prefix} Loss {test_losses[f'{prefix}_total_loss'].item()/i}")

        # post-process losses
        for k, v in test_losses.items():
            if torch.cuda.device_count() > 1:
                dist.all_reduce(test_losses[k])
            test_losses[k] = test_losses[k]/torch.cuda.device_count()

        # post-process ious
        for k, v in test_ious.items():
            if torch.cuda.device_count() > 1:
                dist.all_reduce(test_ious[k])
            test_ious[k] = test_ious[k].cpu()

        iu = test_ious['intersection_sum'] / (test_ious['union_sum'] + 1e-12)
        for ind in range(self.c.model_params.output_classes):
            test_losses[f'test_iou/{prefix}_iou_{ind}'] = float(iu[ind])

        mean_iu = torch.mean(iu)
        test_losses[f'{prefix}_miou'] = float(mean_iu)

        return test_losses

    def process_ious_batch(self, vis):
        pred = torch.argmax(vis['logits'], dim=1)
        intersection, union = self.intersectionAndUnionGPU(
            pred, vis['y'].cuda(), self.c.model_params.output_classes)
        ious = {}
        ious['intersection_sum'] = intersection
        ious['union_sum'] = union
        return ious

    def intersectionAndUnionGPU(self, output, target, K, ignore_index=19):
        # from https://github.com/hszhao/semseg/blob/master/util/util.py
        # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
        assert (output.dim() in [1, 2, 3])
        assert output.shape == target.shape
        output = output.view(-1)
        target = target.view(-1)
        output[target == ignore_index] = ignore_index
        intersection = output[output == target]
        # https://github.com/pytorch/pytorch/issues/1382
        area_intersection = torch.histc(
            intersection.float(), bins=K, min=0, max=K-1)
        area_output = torch.histc(output.float(), bins=K, min=0, max=K-1)
        area_target = torch.histc(target.float(), bins=K, min=0, max=K-1)
        area_union = area_output + area_target - area_intersection
        return area_intersection, area_union
