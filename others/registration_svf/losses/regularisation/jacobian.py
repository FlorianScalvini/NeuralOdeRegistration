import torch
import torch.nn as nn
from typing import Union



def compute_jacobian_determinant_3d(displacement, spacing=(1.0, 1.0, 1.0)):
    """
    Compute the Jacobian determinant of a 3D displacement field.

    Parameters:
    - displacement: torch.Tensor of shape (3, D, H, W), representing the displacement field.
    - spacing: tuple of floats (dz, dy, dx), representing the physical spacing between voxels.

    Returns:
    - jacobian_determinant: torch.Tensor of shape (D, H, W), the Jacobian determinant at each point.
    """
    displacement = displacement.squeeze(0)

    dz, dy, dx = spacing
    grads = []
    for i in range(3):  # u_x, u_y, u_z
        grad_i = torch.gradient(displacement[i], spacing=(dz, dy, dx), dim=(0, 1, 2))
        grads.append(grad_i)

    jacobian = torch.stack([torch.stack(grad_k, dim=-1) for grad_k in grads], dim=-1)
    # Add identity to convert ∂φ/∂x = I + ∂u/∂x
    identity = torch.eye(3).to(displacement.device)
    jacobian = jacobian + identity
    det_j = torch.linalg.det(jacobian)
    return det_j.unsqueeze(0)



def compute_jacobian_determinant(J):
    # Assume J is in normalized coordinates [-1, 1]
    # Step 1: Normalize from [-1, 1] → [0, 1]
    J = J + 1          #恢复到0到1
    J = J / 2.
    scale_factor = torch.tensor([J.size(1), J.size(2), J.size(3)]).to(J).view(1, 1, 1, 1, 3) * 1.
    J = J * scale_factor

    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
    Jdet1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 0])
    Jdet2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy[:, :, :, :, 1] * dz[:, :, :, :, 0])

    Jdet = Jdet0 - Jdet1 + Jdet2
    return Jdet


def jacobian_determinant_3d(deformed_grid: torch.Tensor) -> torch.Tensor:
    """
    Computes the determinant of the Jacobian numerically, given the deformed
    output grid and returns the percentage of negative values

    Args:
        deformed_grid (torch.Tensor): [B, D, H, W, 3]

    Returns:
        torch.Tensor: the percentage of negative determinants
    """
    dy = deformed_grid[:, 1:, :-1, :-1, :] - deformed_grid[:, :-1, :-1, :-1, :]
    dx = deformed_grid[:, :-1, 1:, :-1, :] - deformed_grid[:, :-1, :-1, :-1, :]
    dz = deformed_grid[:, :-1, :-1, 1:, :] - deformed_grid[:, :-1, :-1, :-1, :]

    det0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
    det1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 0])
    det2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy[:, :, :, :, 1] * dz[:, :, :, :, 0])

    determinants = det0 - det1 + det2

    return determinants


class Jacobianloss(nn.Module):
    """
    Jacobian loss for penalizing the Jacobian determinant of a displacement field.
    This loss can be used to ensure that the transformation is invertible and smooth.
    """
    def __init__(self):
        super(Jacobianloss, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Penalizing Jacobian
        Args:
            x (torch.Tensor): Displacement field of shape (B, C, D, H, W) or (B, C, H, W).
            spacing (Union[list[int], tuple[int]]): Spacing of the input image.
        Returns:
            torch.Tensor: Jacobian loss value.
        '''

        Jdet = compute_jacobian_determinant(x)
        Neg_Jac = 0.5 * (torch.abs(Jdet) - Jdet)
        return torch.sum(Neg_Jac)
