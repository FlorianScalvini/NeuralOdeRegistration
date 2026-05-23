# Derived of the DynUnet's Monai Implementation
from typing import Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn



class UnetResBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        dropout: tuple | str | float | None = None,
    ):
        super().__init__()
        layer = []
        layer.append(nn.Conv3d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride, bias=True, padding=kernel_size // 2 if isinstance(kernel_size, int) else [k // 2 for k in kernel_size]))
        if dropout is not None:
            layer.append(nn.Dropout3d(p=dropout, inplace=True))
        self.conv1 = nn.Sequential(*layer)
        layer = []
        layer.append(nn.Conv3d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, bias=True,  padding=kernel_size // 2 if isinstance(kernel_size, int) else [k // 2 for k in kernel_size]))
        if dropout is not None:
            layer.append(nn.Dropout3d(p=dropout, inplace=True))
        self.conv2 = nn.Sequential(*layer)
        self.lrelu= nn.LeakyReLU(inplace=True, negative_slope=0.01)
        self.norm1 = nn.InstanceNorm3d(mid_channels, affine=True)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=True),
                nn.InstanceNorm3d(out_channels, affine=True),
            )

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.lrelu(out)
        return out


class UnetBasicBlock(nn.Module):
    """
    A CNN module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        dropout: dropout probability.

    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        dropout: tuple | str | float | None = None,
    ):
        super().__init__()
        layer = []
        pad = kernel_size // 2 if isinstance(kernel_size, int) else [k // 2 for k in kernel_size]
        layer.append(nn.Conv3d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride, bias=True, padding=pad))
        if dropout is not None:
            layer.append(nn.Dropout3d(p=dropout, inplace=True))
        self.conv1 = nn.Sequential(*layer)
        layer = []
        layer.append(nn.Conv3d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, bias=True, padding=pad))
        if dropout is not None:
            layer.append(nn.Dropout3d(p=dropout, inplace=True))
        self.conv2 = nn.Sequential(*layer)
        self.lrelu= nn.LeakyReLU(inplace=True, negative_slope=0.01)
        self.norm1 = nn.InstanceNorm3d(mid_channels, affine=True)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        return out


class UnetUpBlock(nn.Module):
    """
    An upsampling module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        dropout: tuple | str | float | None = None,
        upsample_conv : bool = True,
    ):
        super().__init__()
        layer = []

        if upsample_conv:
            layer.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=1, padding=kernel_size // 2 if isinstance(kernel_size, int) else [k // 2 for k in kernel_size], bias=True))
            layer.append(nn.InstanceNorm3d(out_channels, affine=True))
            layer.append(nn.LeakyReLU(inplace=True, negative_slope=0.01))
        layer.append(nn.Upsample(scale_factor=2.0, mode='trilinear', align_corners=True))
        self.upsample = nn.Sequential(*layer)
        self.conv_block = UnetBasicBlock(
            in_channels=out_channels + out_channels,
            mid_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
        )

    def forward(self, inp, skip):
        out = self.upsample(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)

        return out


class UnetOutBlock(nn.Module):

    def __init__(
        self, in_channels: int, out_channels: int, dropout: float | None = None):
        super().__init__()
        layers = []
        layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=True))
        if dropout is not None:
            layers.append(nn.Dropout3d(p=dropout, inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, inp):
        return self.conv(inp)

class DyNUnet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[Sequence[int]] = None,
        strides: Sequence[Sequence[int]] = None,
        filters: Sequence[int] = None,
        dropout: Optional[Union[Tuple, str, float]] = None,
        res_block: bool = False,
    ):
        self.conv_block = UnetResBlock if res_block else UnetBasicBlock
        super().__init__()
        self.drop_out = dropout
        self.kernel_size = kernel_size
        self.strides = strides



        self.kernel_size = kernel_size
        self.strides = strides
        if filters is None:
            self.filters = [min(2 ** (5 + i), 320) for i in range(len(self.strides))]

        self.input_block = self.conv_block(in_channels=in_channels, mid_channels=self.filters[0], out_channels=self.filters[0],
                                           stride=1, kernel_size=[3, 3, 3], dropout=self.drop_out)
        self.down_blocks = nn.ModuleList([
            self.conv_block(in_channels=self.filters[i], mid_channels=self.filters[i + 1], out_channels=self.filters[i + 1],
                            stride=self.strides[i], kernel_size=self.kernel_size[i], dropout=self.drop_out) for i in range(len(self.filters) - 1)
        ])

        self.bottleneck = self.conv_block(in_channels=self.filters[-1], mid_channels=self.filters[-1], out_channels=self.filters[-1], stride=self.strides[-1], kernel_size=self.kernel_size[-1], dropout=self.drop_out)

        self.up_block = nn.ModuleList([
            UnetUpBlock(in_channels=self.filters[-1], out_channels=self.filters[-1], kernel_size=[1, 1, 1], dropout=self.drop_out, upsample_conv=False)
        ])
        for i in range(len(self.filters) - 1, 0, -1):
            self.up_block.append(
                UnetUpBlock(in_channels=self.filters[i], out_channels=self.filters[i - 1],
                            kernel_size=[1, 1, 1], dropout=self.drop_out))
        self.output_block = UnetOutBlock(in_channels=self.filters[0],out_channels=out_channels,dropout=self.drop_out)
        self.apply(self.initialize_weights)



    def forward(self, images):
        skips = [self.input_block(images)]
        for i in range(len(self.down_blocks)):
            skips.append(self.down_blocks[i](skips[-1]))
        y = self.bottleneck(skips[-1])
        for i, up in enumerate(self.up_block):
            y = up(y, skips[-(i + 1)])
        y = self.output_block(y)
        return y


    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


