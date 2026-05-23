import torch
from torch import Tensor
import torch.nn as nn
import monai


class RegistrationModule(nn.Module):
    '''
        Registration module for 3D image registration.yaml with DVF
    '''
    def __init__(self, model, int_steps=7):
        '''
        :param model: nn.Module
        '''
        super().__init__()
        self.model = model
        self.dvf2ddf = monai.networks.blocks.DVF2DDF(num_steps=int_steps, mode='bilinear', padding_mode='zeros')  # Vector integration based on Runge-Kutta method

    def forward(self, data: Tensor) -> Tensor:
        '''
            Forward pass of the registration module
            :param data: Input images
            :return: Deformation field
        '''
        y = self.model(data)
        return y

    def load_network(self, path) -> None:
        '''
            Load the network weights
            :param path: Path to the weights
        '''
        self.model.load_state_dict(torch.load(path))

    def velocity2displacement(self, dvf: Tensor) -> Tensor:
        '''
            Convert the velocity field to a flow field
            :param dvf: Velocity field
            :return: Deformation field
        '''
        return self.dvf2ddf(dvf)