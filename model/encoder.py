import math
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from torch import FloatTensor, LongTensor

from .pos_enc import ImageRotaryEmbed, ImgPosEnc


# DenseNet-B
class _Bottleneck(nn.Module):
    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool):
        super(_Bottleneck, self).__init__()
        interChannels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(n_channels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate)
        self.conv2 = nn.Conv2d(
            interChannels, growth_rate, kernel_size=3, padding=1, bias=False
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# single layer
class _SingleLayer(nn.Module):
    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool):
        super(_SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(
            n_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv1(F.relu(x, inplace=True))
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# transition layer
class _Transition(nn.Module):
    def __init__(self, n_channels: int, n_out_channels: int, use_dropout: bool):
        super(_Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_out_channels)
        self.conv1 = nn.Conv2d(n_channels, n_out_channels, kernel_size=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True)
        return out


class ResBlock(nn.Module):
    def __init__(self,
                 input_channels: int,
                 num_channels: int,
                 use_1x1conv: bool = False,
                 strides: int = 1,
                 use_dropout: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))

        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        if self.use_dropout:
            y = self.dropout(y)
        y += x
        return F.relu(y)


class ResNet18(nn.Module):
    def __init__(
            self,
            use_dropout: bool = True
    ):
        super(ResNet18, self).__init__()
        b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm2d(64), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # 经过会降采样
        b2 = nn.Sequential(*self._make_res(64, 64, 2, first_block=True, use_dropout=use_dropout))
        b3 = nn.Sequential(*self._make_res(64, 128, 2, use_dropout=use_dropout))
        b4 = nn.Sequential(*self._make_res(128, 256, 2, use_dropout=use_dropout))
        b5 = nn.Sequential(*self._make_res(256, 512, 2, use_dropout=use_dropout))
        self.net = nn.Sequential(b1, b2, b3, b4, b5)  # (batch_size, 512, 5次降采样的h*w)

    @staticmethod
    def _make_res(input_channels, num_channels, num_residuals, first_block=False, use_dropout=True):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(ResBlock(input_channels, num_channels,
                                    use_1x1conv=True, strides=2, use_dropout=use_dropout))
            else:
                blk.append(ResBlock(num_channels, num_channels, use_dropout=use_dropout))
        return blk

    def forward(self, x, x_mask):

        out_mask = x_mask[:, 0::2, 0::2]  # mask随图像的降采样而相应降采样，主要由stride=2导致
        out_mask = out_mask[:, 0::2, 0::2]  # mask随图像的降采样而相应降采样，主要由stride=2导致
        out_mask = out_mask[:, 0::2, 0::2]
        out_mask = out_mask[:, 0::2, 0::2]
        out_mask = out_mask[:, 0::2, 0::2]
        out = self.net(x)
        return out, out_mask


# resnet50
class ConvBlock(nn.Module):
    def __init__(self, in_channel, f, filters, s):
        super(ConvBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, F1, 1, stride=s, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1, F2, f, stride=1, padding=True, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.shortcut_1 = nn.Conv2d(in_channel, F3, 1, stride=s, padding=0, bias=False)
        self.batch_1 = nn.BatchNorm2d(F3)
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


class IndentityBlock(nn.Module):
    def __init__(self, in_channel, f, filters):
        super(IndentityBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, F1, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1, F2, f, stride=1, padding=True, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


class ResModel(nn.Module):
    def __init__(self):
        super(ResModel, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(64, f=3, filters=[64, 64, 128], s=1),
            IndentityBlock(128, 3, [64, 64, 128]),
            IndentityBlock(128, 3, [64, 64, 128]),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(128, f=3, filters=[128, 128, 128], s=2),
            IndentityBlock(128, 3, [128, 128, 128]),
            IndentityBlock(128, 3, [128, 128, 128]),
            IndentityBlock(128, 3, [128, 128, 128]),
        )
        self.stage4 = nn.Sequential(
            ConvBlock(128, f=3, filters=[256, 256, 256], s=2),
            IndentityBlock(256, 3, [256, 256, 256]),
            IndentityBlock(256, 3, [256, 256, 256]),
            IndentityBlock(256, 3, [256, 256, 256]),
            IndentityBlock(256, 3, [256, 256, 256]),
            IndentityBlock(256, 3, [256, 256, 256]),
        )
        self.stage5 = nn.Sequential(
            ConvBlock(256, f=3, filters=[512, 512, 512], s=2),
            IndentityBlock(512, 3, [512, 512, 512]),
            IndentityBlock(512, 3, [512, 512, 512]),
        )
        self.pool = nn.AvgPool2d(3, 1, padding=1)
        # self.fc = nn.Sequential(
        #     nn.Linear(8192, n_class)
        # )

    def forward(self, X, x_mask):
        out = self.stage1(X)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.pool(out)
        out_mask = x_mask[:, 0::2, 0::2]  # mask随图像的降采样而相应降采样，主要由stride=2导致
        out_mask = out_mask[:, 0::2, 0::2]  # mask随图像的降采样而相应降采样，主要由stride=2导致
        out_mask = out_mask[:, 0::2, 0::2]
        out_mask = out_mask[:, 0::2, 0::2]
        # out_mask = out_mask[:, 0::2, 0::2]
        # out = out.view(out.size(0), 8192)
        # out = self.fc(out)
        return out, out_mask


class DenseNet(nn.Module):  # 按照lightening库规范后的网络，共包含dense块3个，过渡层2个
    def __init__(
            self,
            growth_rate: int,
            num_layers: int,
            reduction: float = 0.5,
            bottleneck: bool = True,
            use_dropout: bool = True,
    ):
        super(DenseNet, self).__init__()
        n_dense_blocks = num_layers  # 一个dense_block内卷积层的个数，
        n_channels = 2 * growth_rate  # growth_rate是一个block内卷积层的通道数
        self.conv1 = nn.Conv2d(
            1, n_channels, kernel_size=7, padding=3, stride=2, bias=False  # 从1通道到两倍的groth_rate
        )  # H和W变为原来的一半
        self.norm1 = nn.BatchNorm2d(n_channels)  # 通道维度BN化
        self.dense1 = self._make_dense(
            n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout  # 创建dense块，选择使用bottleneck和dropout技术
        )
        n_channels += n_dense_blocks * growth_rate  # 通道数增加，下边的代码写的不好，可以使用循环优化
        n_out_channels = int(math.floor(n_channels * reduction))  # 为使用过渡层提供输出通道数量，默认为原来的一半
        self.trans1 = _Transition(n_channels, n_out_channels, use_dropout)

        n_channels = n_out_channels
        self.dense2 = self._make_dense(
            n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout
        )
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans2 = _Transition(n_channels, n_out_channels, use_dropout)

        n_channels = n_out_channels
        self.dense3 = self._make_dense(
            n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout
        )

        self.out_channels = n_channels + n_dense_blocks * growth_rate
        self.post_norm = nn.BatchNorm2d(self.out_channels)

    @staticmethod
    def _make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout):
        layers = []
        for _ in range(int(n_dense_blocks)):
            if bottleneck:
                layers.append(_Bottleneck(n_channels, growth_rate, use_dropout))
            else:
                layers.append(_SingleLayer(n_channels, growth_rate, use_dropout))
            n_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, x_mask):
        out = self.conv1(x)
        out = self.norm1(out)
        out_mask = x_mask[:, 0::2, 0::2]  # mask随图像的降采样而相应降采样，主要由stride=2导致
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, 2, ceil_mode=True)
        out_mask = out_mask[:, 0::2, 0::2]  # mask随图像的降采样而相应降采样，主要由stride=2导致
        out = self.dense1(out)
        out = self.trans1(out)
        out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense2(out)
        out = self.trans2(out)
        out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense3(out)
        out = self.post_norm(out)
        return out, out_mask


class Encoder(pl.LightningModule):
    def __init__(self, d_model: int, growth_rate: int, num_layers: int):
        super().__init__()
        # num_layers是指densenet一个block内的卷积层的个数，denseblock个数固定为了3
        # self.model = DenseNet(growth_rate=growth_rate, num_layers=num_layers)
        # self.model = ResNet18(use_dropout=True)
        self.model = ResModel()
        self.feature_proj = nn.Sequential(
            nn.Conv2d(512, d_model, kernel_size=1),  # out_channels是densenet输出的总通道数
            nn.ReLU(inplace=True),  # 通过此层将通道数限制到超参数d_model维
        )
        self.norm = nn.LayerNorm(d_model)

        self.pos_enc_2d = ImgPosEnc(d_model, normalize=True)  # 实例化图像位置编码类

    def forward(
            self, img: FloatTensor, img_mask: LongTensor
    ) -> Tuple[FloatTensor, LongTensor]:
        """encode image to feature

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']

        Returns
        -------
        Tuple[FloatTensor, LongTensor]
            [b, t, d], [b, t]
        """
        # extract feature
        feature, mask = self.model(img, img_mask)  # 经过densenet编码的图像和mask，有大幅度的降采样
        feature = self.feature_proj(feature)  # 仅限制输出的通道数

        # proj
        feature = rearrange(feature, "b d h w -> b h w d")  # 通道维度换到了最后，为位置编码铺路
        feature = self.norm(feature)

        # positional encoding
        feature = self.pos_enc_2d(feature, mask)

        # flat to 1-D
        feature = rearrange(feature, "b h w d -> b (h w) d")
        mask = rearrange(mask, "b h w -> b (h w)")  # 全部将像素拉平
        return feature, mask


if __name__ == '__main__':
    x = torch.rand(2, 1, 280, 90)
    x_m = torch.rand(2, 280, 90)
    # model = ResNet18()
    model = ResModel()
    y = model(x, x_m)
    print(y[0].shape, y[1].shape)
