from monai.networks.blocks import ResidualUnit, Convolution
from torch import nn
import torch.nn.functional as F
import torch
from typing import Sequence

class Conv3dReLU(nn.Sequential):
    def __init__(self,in_channels: int, out_channels: int, kernel_size: Sequence[int] | int, padding=0, stride=1):
        super().__init__()
        self.conv = nn.Conv3d( in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.relu = nn.LeakyReLU(inplace=True)
        self.nm = nn.InstanceNorm3d(out_channels)

    def forward(self, x) -> torch.Tensor:
        out = self.conv(x)
        out = self.nm(out)
        out = self.relu(out)
        return out

class UnetBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int,  t_dim, kernel_size: Sequence[int] | int, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(UnetBlock, self).__init__()
        self.cvbact_1 = Conv3dReLU(in_channels, mid_channels, kernel_size, padding=padding, stride=stride)
        self.cvbact_2 = Conv3dReLU(mid_channels, out_channels, kernel_size, padding=padding, stride=1)
        self.time_mlp = nn.Linear(t_dim, out_channels * 2, bias=True)
        self.spatial_dims = 3

    def forward(self, x, t) -> torch.Tensor:

        t_embed = self.time_mlp(t)
        spatial_shape = [1] * self.spatial_dims
        out = self.cvbact_1(x)
        out = self.cvbact_2(out)
        gamma, beta = t_embed.chunk(2, dim=-1)
        gamma = gamma.view(*gamma.shape, *spatial_shape)
        beta = beta.view(*beta.shape, *spatial_shape)
        out = out * gamma + beta
        return out

class UnetUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Sequence[int] | int, t_dim):
        super().__init__()
        self.upsample = nn.Sequential(
            Conv3dReLU(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2 if isinstance(kernel_size, int) else [k // 2 for k in kernel_size]),
            nn.Upsample(scale_factor=2.0, mode='trilinear', align_corners=True)
        )
        self.conv_block = Conv3dReLU(out_channels * 2, out_channels, kernel_size=1)

        self.spatial_dims = 3
        self.time_mlp = nn.Linear(t_dim, out_channels * 2, bias=True)

    def forward(self, x, t, x_skip):
        t_embed = self.time_mlp(t)
        spatial_shape = [1] * self.spatial_dims
        out = self.upsample(x)
        out = torch.cat((out, x_skip), dim=1)
        out = self.conv_block(out)
        gamma, beta = t_embed.chunk(2, dim=-1)
        gamma = gamma.view(*gamma.shape, *spatial_shape)
        beta = beta.view(*beta.shape, *spatial_shape)
        out = out * gamma + beta
        return out

class UnetUpBlockNoSkip(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Sequence[int] | int, t_dim):
        super().__init__()
        self.upsample = nn.Sequential(
            Conv3dReLU(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2 if isinstance(kernel_size, int) else [k // 2 for k in kernel_size]),
            nn.Upsample(scale_factor=2.0, mode='trilinear', align_corners=True)
        )
        self.conv_block = Conv3dReLU(out_channels, out_channels, kernel_size=1)

        self.spatial_dims = 3
        self.time_mlp = nn.Linear(t_dim, out_channels, bias=True)

    def forward(self, x, t):
        t_embed = self.time_mlp(t)
        spatial_shape = [1] * self.spatial_dims
        out = self.upsample(x)
        out = self.conv_block(out)
        out = out + t_embed.view(1, -1, 1, 1 ,1)
        return out

class EncoderUnet(nn.Module):
    def __init__(self, in_channels: int, channels: Sequence[int], t_dim) -> None:
        super().__init__()
        ch = [in_channels] + list(channels)
        self.encoder = nn.ModuleList([
            UnetBlock(ch[i], ch[i+1], ch[i+1],
                      kernel_size=3, padding=1,
                      stride=1 if i == 0 else 2,
                      t_dim=t_dim)
            for i in range(len(channels))
        ])

    def forward(self, x: torch.Tensor, t) -> Sequence[torch.Tensor]:
        feats = []
        for stage in self.encoder:
            x = stage(x, t)
            feats.append(x)
        return feats



class FiLM(nn.Module):
    """
    Predicts per-channel scale γ and shift β from z,
    then applies: γ * x + β  (element-wise on feature map x).

    Args:
        z_dim    : dimension of the context vector
        n_chan   : number of channels in the feature map to modulate
    """
    def __init__(self, z_dim: int, n_chan: int):
        super().__init__()
        self.proj = nn.Linear(z_dim, 2 * n_chan)
        # Init γ ≈ 1, β ≈ 0  → identity at start of training
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias[:n_chan],  1.0)  # γ
        nn.init.constant_(self.proj.bias[n_chan:],  0.0)  # β

    def forward(self,
                x: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W, D)
        z : (B, z_dim)
        """
        gamma, beta = self.proj(z).chunk(2, dim=-1)   # (B, C) each
        gamma = gamma.view(-1, x.shape[1], 1, 1, 1)
        beta  = beta.view(-1,  x.shape[1], 1, 1, 1)
        return gamma * x + beta
