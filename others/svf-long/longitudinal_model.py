import os
from registration_svf.registration import RegistrationModule
from registration_svf.modules.monotonic_mlp import MonotonicMLP
import torch
import torch.nn as nn


class LongitudinalDeformation(nn.Module):
    def __init__(self, svf_model : RegistrationModule, time_mode: str, t0: int, t1: int):
        '''
        Our longitudinal deformation model
        :param svf_model: Registration model
        :param time_mode: Interpolation mode, either 'mlp', 'inr' or 'linear'
        :param t0: time 0
        :param t1: time 1
        :param size: INR size

        '''
        super().__init__()
        self.t0 = t0
        self.t1 = t1
        self.svf_model = svf_model
        self.time_mode = time_mode
        self.mlp_model = None
        if self.time_mode == 'mlp':
            self.mlp_model = MonotonicMLP()

    def forward(self, data : torch.Tensor) -> torch.Tensor:
        return self.svf_model(data)

    def encode_time(self, time: torch.Tensor) -> torch.Tensor:
        if self.time_mode == 'mlp':
            time =  self.mlp_model(time)
        return time

    def load_reg_model(self, path) -> None:
        self.svf_model.load_state_dict(torch.load(path))

    def load_temporal(self, path) -> None:
        if self.mlp_model is not None:
            self.mlp_model.load_state_dict(torch.load(path))

    def save(self, path) -> None:
        torch.save(self.svf_model.state_dict(), os.path.join(path, 'model.pth'))
        if self.mlp_model is not None:
            torch.save(self.mlp_model.state_dict(), os.path.join(path, 'temporal_model.pth'))


