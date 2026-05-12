from typing import Any, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import pytorch_lightning as pl
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
import torch.utils.checkpoint as checkpoint
import monai
from torch.nn.utils import spectral_norm as sn
from torch import Tensor
import matplotlib.pyplot as plt
from PIL import Image
import io

def warp(image: Tensor, flow: Tensor, mode: str = 'bilinear') -> Tensor:
    warper = monai.networks.blocks.Warp(mode=mode, padding_mode='reflection')
    return warper(image, flow)


def generate_grid3d_tensor(shape):
    x = torch.linspace(-1., 1., shape[0])
    y = torch.linspace(-1., 1., shape[1])
    z = torch.linspace(-1., 1., shape[2])
    x, y, z = torch.meshgrid(x, y, z, indexing='ij')
    return torch.stack([z, y, x], dim=0)   # (3, D,



# ──────────────────────────────────────────────────────────────
# Sinusoidal embedding  (for scalar time T and t₀)
# ──────────────────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or scalar  →  (B, dim)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1)
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)   # (B, dim)



# ──────────────────────────────────────────────────────────────
# 3D Sinusoidal positional encoding
# ──────────────────────────────────────────────────────────────

class SinPositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(SinPositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels/6)*2)
        if channels % 2:
            channels += 1
        self.channels = channels
        self.inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        #self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        tensor = tensor.permute(0, 2, 3, 4, 1)
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x,y,z,self.channels*3),device=tensor.device).type(tensor.type())
        emb[:,:,:,:self.channels] = emb_x
        emb[:,:,:,self.channels:2*self.channels] = emb_y
        emb[:,:,:,2*self.channels:] = emb_z
        emb = emb[None,:,:,:,:orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return emb.permute(0, 4, 1, 2, 3)


# ──────────────────────────────────────────────────────────────
# Adaptive LayerNorm  (adaLN)
# ──────────────────────────────────────────────────────────────

class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * dim)   # ← FIXED: cond_dim not dim

    def forward(self, x, cond):
        gamma, beta = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1 + gamma) + beta

class Grad3d(nn.Module):
    def __init__(self, penalty='l1'):
        super().__init__()
        if penalty not in ['l1', 'l2']:
            raise ValueError(f"Unknown penalty type: {penalty}")
        self.penalty = penalty

    def forward(self, x_pred):
        # Compute gradients in each direction
        dx = torch.abs(x_pred[:, :, 1:, :, :] - x_pred[:, :, :-1, :, :])
        dy = torch.abs(x_pred[:, :, :, 1:, :] - x_pred[:, :, :, :-1, :])
        dz = torch.abs(x_pred[:, :, :, :, 1:] - x_pred[:, :, :, :, :-1])
        # Apply penalty (squared for L2 penalty)
        if self.penalty == 'l2':
            dy, dx, dz = dy**2, dx**2, dz**2
        grad = (torch.mean(dx) + torch.mean(dy) + torch.mean(dz)) / 3.0
        return grad


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, L, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, window_size, C)
    """
    B, H, W, L, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], L // window_size[2], window_size[2], C)

    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows

def normalize_to_0_1(volume):
    '''
        Normalize volume to 0-1 range
    '''
    max_val = volume.max()
    min_val = volume.min()
    return (volume - min_val) / (max_val - min_val)


def window_reverse(windows, window_size, H, W, L):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
        L (int): Length of image
    Returns:
        x: (B, H, W, L, C)
    """
    B = int(windows.shape[0] / (H * W * L / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, H // window_size[0], W // window_size[1], L // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, L, -1)
    return x

def get_reference_grid(ddf: torch.Tensor) -> torch.Tensor:
    mesh_points = [torch.arange(0, dim) for dim in ddf.shape[2:]]
    grid = torch.stack(meshgrid_ij(*mesh_points), dim=0)  # (spatial_dims, ...)
    grid = torch.stack([grid] * ddf.shape[0], dim=0)  # (batch, spatial_dims, ...)
    ref_grid = grid.to(ddf)
    return ref_grid


def meshgrid_ij(*tensors):
    if torch.meshgrid.__kwdefaults__ is not None and "indexing" in torch.meshgrid.__kwdefaults__:
        return torch.meshgrid(*tensors, indexing="ij")  # new api pytorch after 1.10
    return torch.meshgrid(*tensors)

def displacement2grid(flow: Tensor) -> torch.Tensor:
    """
    Convert a flow field to a normalized sampling grid (phi) for grid_sample.

    Args:
        :param flow : Tensor - Flow field of shape (B, 3, D, H, W)

    Returns:
        Tensor: Normalized grid (phi) suitable for F.grid_sample.
        :param grid_normalize:
    """
    spatial_dims = len(flow.shape) - 2
    if spatial_dims not in (2, 3):
        raise NotImplementedError(f"got unsupported spatial_dims={spatial_dims}, currently support 2 or 3.")
    grid = get_reference_grid(flow).to(flow.device) + flow

    grid = grid.permute([0] + list(range(2, 2 + spatial_dims)) + [1])
    for i, dim in enumerate(grid.shape[1:-1]):
        grid[..., i] = grid[..., i] * 2 / (dim - 1) - 1
    return grid


def plt_grid(xy: torch.Tensor, ratio=0.8):
    """
    Plots the 2D grid
    Args:
        xy (torch.Tensor): generated grids [h, w]
    """

    # Figure size in inches = pixels / DPI
    dpi = 100
    width_px = int(1000 * ratio)
    height_px = 1000

    # Create figure with exact size
    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # Fill entire canvas

    # Set black background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Draw grid lines in white
    for i in range(xy.shape[0]):
        ax.plot(xy[i, :, 0], xy[i, :, 1], '-', lw=1.3, color='k')
    for j in range(xy.shape[1]):
        ax.plot(xy[:, j, 0], xy[:, j, 1], '-', lw=1.3, color='k')

    ax.set_xlim(xy[..., 0].min(), xy[..., 0].max())
    ax.set_ylim(xy[..., 1].min(), xy[..., 1].max())
    ax.invert_yaxis()  # Optional, depends on how your grid is laid out

    # Hide all axis decorations
    ax.axis('off')

    # Save to a PIL Image using in-memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches=None, pad_inches=0, transparent=False)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    buf.close()

    return image, fig

