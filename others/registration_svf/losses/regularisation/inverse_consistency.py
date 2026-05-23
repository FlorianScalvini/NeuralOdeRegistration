import torch
import torch.nn as nn
from monai.networks.blocks.warp import Warp


class InverseConsistency(nn.Module):
    def __init__(self):
        super().__init__()
        self.warp = Warp()

    def forward(self, phi_a, phi_b):
        identity = self.identity_map(phi_a.shape).to(phi_a.device)
        appro_for = (self.warp(phi_a, phi_b) + phi_b) - identity
        appro_inv = (self.warp(phi_b, phi_a) + phi_a) - identity
        return  torch.mean(appro_for**2) + torch.mean(appro_inv**2)

    @staticmethod
    def identity_map(size) -> torch.Tensor:
        """Creates an identity mapping tensor with spatial dimensions."""
        dim = len(size) - 2  # Number of spatial dimensions
        if dim != size[1]:
            raise ValueError(
                f"Number of spatial dimensions (size[2:]) should match the "
                f"second dimension (channels), got {dim} != {size[1]}"
            )

        # Compute normalized coordinate values
        grids = [torch.linspace(0, 1, steps=size[i + 2]) for i in range(dim)]
        mesh = torch.meshgrid(*grids, indexing='ij')  # Ensures correct indexing
        coordinates = torch.stack(mesh, dim=0)  # Shape: (dim, *size[2:])
        id_map = coordinates.unsqueeze(0).expand(size[0], -1, *[-1] * dim).clone()

        return id_map.float()

class IconInverseConsistency(InverseConsistency):
    def __init__(self):
        super().__init__()

    def forward(self, phi_a, phi_b):
        identity = self.identity_map(phi_a.shape)
        eps_identity = (identity + torch.randn(*identity.shape) / identity.shape[-1]).to(phi_a.device)
        approx_ab = self.warp(eps_identity, self.warp(phi_a, phi_b) + phi_b)
        approx_ba = self.warp(eps_identity, self.warp(phi_b, phi_a) + phi_a)
        loss = (torch.mean((eps_identity - approx_ab) ** 2)  + torch.mean((eps_identity - approx_ba) ** 2))
        return loss


class GradIconInverseConsistency(InverseConsistency):
    def __init__(self):
        super().__init__()
        self.delta = 0.001

    def forward(self, phi_a, phi_b):
        identity = self.identity_map(phi_a.shape)
        eps_identity = (identity + torch.randn(*identity.shape) / identity.shape[-1]).to(phi_a.device)
        approx_ba = self.warp(eps_identity, self.warp(phi_b, phi_a) + phi_a)
        inverse_consistency_error = eps_identity - approx_ba

        if len(identity.shape) == 4:
            dx = torch.Tensor([[[[self.delta]], [[0.0]]]]).to(phi_a.device)
            dy = torch.Tensor([[[[0.0]], [[self.delta]]]]).to(phi_a.device)
            direction_vectors = (dx, dy)
        else:
            dx = torch.Tensor([[[[[self.delta]]], [[[0.0]]], [[[0.0]]]]]).to(phi_a.device)
            dy = torch.Tensor([[[[[0.0]]], [[[self.delta]]], [[[0.0]]]]]).to(phi_a.device)
            dz = torch.Tensor([[[[[0.0]]], [[[0.0]]], [[[self.delta]]]]]).to(phi_a.device)
            direction_vectors = (dx, dy, dz)

        direction_losses = []
        for d in direction_vectors:
            approx_ba_d = self.warp(eps_identity + d, self.warp(phi_b, phi_a) + phi_a)
            inverse_consistency_error_d = eps_identity + d - approx_ba_d
            grad_d_icon_error = (inverse_consistency_error - inverse_consistency_error_d) / self.delta
            direction_losses.append(torch.mean(grad_d_icon_error ** 2))
        loss = sum(direction_losses)
        return loss
