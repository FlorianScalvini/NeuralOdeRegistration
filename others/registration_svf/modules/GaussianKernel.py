
import torch
import numpy as np
import torch.nn.functional as F
import scipy.stats as st

class GaussianKernel(torch.nn.Module):
    def __init__(self, win=11, nsig=0.1):
        super(GaussianKernel, self).__init__()
        self.win = win
        self.nsig = nsig
        kernel_x, kernel_y, kernel_z = self.gkern1D_xyz(self.win, self.nsig)
        kernel = kernel_x * kernel_y * kernel_z
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)
        self.register_buffer("kernel_z", kernel_z)
        self.register_buffer("kernel", kernel)

    def gkern1D(self, kernlen=None, nsig=None):
        '''
        :param nsig: large nsig gives more freedom(pixels as agents), small nsig is more fluid.
        :return: Returns a 1D Gaussian kernel.
        '''
        x = np.linspace(-nsig, nsig, kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kern1d = kern1d / kern1d.sum()
        return torch.tensor(kern1d, requires_grad=False).float()

    def gkern1D_xyz(self, kernlen=None, nsig=None):
        """Returns 3 1D Gaussian kernel on xyz direction."""
        kernel_1d = self.gkern1D(kernlen, nsig)
        kernel_x = kernel_1d.view(1, 1, -1, 1, 1)
        kernel_y = kernel_1d.view(1, 1, 1, -1, 1)
        kernel_z = kernel_1d.view(1, 1, 1, 1, -1)
        return kernel_x, kernel_y, kernel_z

    def forward(self, x):
        pad = int((self.win - 1) / 2)
        # Apply Gaussian by 3D kernel
        x = F.conv3d(x, self.kernel, padding=pad)
        return x

class AveragingKernel(torch.nn.Module):
    def __init__(self, win=11):
        super(AveragingKernel, self).__init__()
        self.win = win

    def window_averaging(self, v):
        win_size = self.win
        v = v.double()

        half_win = int(win_size / 2)
        pad = [half_win + 1, half_win] * 3

        v_padded = F.pad(v, pad=pad, mode='constant', value=0)  # [x+pad, y+pad, z+pad]

        # Run the cumulative sum across all 3 dimensions
        v_cs_x = torch.cumsum(v_padded, dim=2)
        v_cs_xy = torch.cumsum(v_cs_x, dim=3)
        v_cs_xyz = torch.cumsum(v_cs_xy, dim=4)

        x, y, z = v.shape[2:]

        # Use subtraction to calculate the window sum
        v_win = v_cs_xyz[:, :, win_size:, win_size:, win_size:] \
                - v_cs_xyz[:, :, win_size:, win_size:, :z] \
                - v_cs_xyz[:, :, win_size:, :y, win_size:] \
                - v_cs_xyz[:, :, :x, win_size:, win_size:] \
                + v_cs_xyz[:, :, win_size:, :y, :z] \
                + v_cs_xyz[:, :, :x, win_size:, :z] \
                + v_cs_xyz[:, :, :x, :y, win_size:] \
                - v_cs_xyz[:, :, :x, :y, :z]

        # Normalize by number of elements
        v_win = v_win / (win_size ** 3)
        v_win = v_win.float()
        return v_win

    def forward(self, v):
        return self.window_averaging(v)