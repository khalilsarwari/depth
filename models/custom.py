import torch
import torch.nn as nn
import torch.nn.functional as F

from .miniViT import mViT


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=2048):
        super(DecoderBN, self).__init__()
        features = int(num_features)

        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSampleBN(skip_input=features // 1 + 112 + 64, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + 40 + 24, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + 24 + 16, output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + 16 + 8, output_features=features // 16)

        #         self.up5 = UpSample(skip_input=features // 16 + 3, output_features=features//16)
        self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)
        # self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[
            11]

        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        #         x_d5 = self.up5(x_d4, features[0])
        out = self.conv3(x_d4)
        # out = self.act_out(out)
        # if with_features:
        #     return out, features[-1]
        # elif with_intermediate:
        #     return out, [x_block0, x_block1, x_block2, x_block3, x_block4, x_d1, x_d2, x_d3, x_d4]
        return out


class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


class Custom(nn.Module):
    def __init__(self, backend, n_bins=100, min_val=0.1, max_val=10, norm='linear'):
        super(Custom, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.encoder = Encoder(backend)

        self.decoder = DecoderBN(num_classes=64)
        self.bins_out = nn.Sequential(nn.Conv2d(128, n_bins,kernel_size=1, stride=1, padding=0),
                                      nn.ReLU())
        self.conv_out = nn.Sequential(nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))

    def forward(self, x, **kwargs):
        # print(x.shape) -> torch.Size([batchsize, 3, 352, 704])
        unet_out = self.decoder(self.encoder(x), **kwargs) # [batchsize, 128, 176, 352]
        eps = 1e-6
        with torch.no_grad():
            mean = unet_out.mean(dim=[2, 3], keepdim=True)
            std = unet_out.std(dim=[2, 3], keepdim=True)
            # print('unet_out', unet_out.shape)
            # print('unet_out mean', mean.shape)
            # print('unet_out std', std.shape)
            unet_g = (unet_out - mean)/(std + eps)
        unet_out_g = torch.cat([unet_out, unet_g], dim=1)

        y = self.bins_out(unet_out_g)
        y = torch.relu(y)
        y = y + eps # eps was originally .1
        bin_widths_normed = y / y.sum(dim=1, keepdim=True) # pixelwise bin widths

        out = self.conv_out(unet_out_g) # unet out g should be torch.Size([1, 256, 176, 352])

        # Post process
        # n, c, h, w = out.shape
        # hist = torch.sum(out.view(n, c, h * w), dim=2) / (h * w)  # not used for training

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        # print(bin_widths.shape, 'bin_widths')
        bin_widths = nn.functional.pad(bin_widths, (0, 0, 0, 0, 0, 1), mode='constant', value=self.min_val)
        # print(bin_widths.shape, 'bin_widths post paddingg')
        bin_edges = torch.cumsum(bin_widths, dim=1)
        # print(bin_edges.shape, 'bin_edges')

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        # n, dout = centers.size()
        # centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)
        # print(pred.shape) -> torch.Size([1, 1, 176, 352])
        avg_bin_edges = bin_edges.mean(dim=[2, 3]) # convert pixelwise bins to imagewise bins via averaging

        return avg_bin_edges, pred

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.bins_out, self.conv_out]
        for m in modules:
            yield from m.parameters()

    @classmethod
    def build(cls, n_bins, **kwargs):
        basemodel_name = 'tf_efficientnet_b5_ap'

        print('Loading base model ()...'.format(basemodel_name), end='')
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
        print('Done.')

        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print('Building Encoder-Decoder model..', end='')
        m = cls(basemodel, n_bins=n_bins, **kwargs)
        print('Done.')
        return m


if __name__ == '__main__':
    model = UnetAdaptiveBins.build(100)
    x = torch.rand(2, 3, 480, 640)
    bins, pred = model(x)
    print(bins.shape, pred.shape)
