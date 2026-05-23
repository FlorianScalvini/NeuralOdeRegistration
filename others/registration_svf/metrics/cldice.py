import torch
import torch.nn as nn
from skimage.morphology import skeletonize

class clDiceMetric(nn.Module):
    def __init__(self):
        """
        Compute the Soft clDice loss defined in:

            Shit et al. (2021) clDice -- A Novel Topology-Preserving Loss Function
            for Tubular Structure Segmentation. (https://arxiv.org/abs/2003.07311)

        Adapted from:
            https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/cldice.py#L7
        """
        super(clDiceMetric, self).__init__()

    def cl_score(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        skel_pred = cl.soft_skel(y_pred, self.iter)
        skel_true = cl.soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:, 1:, ...]) + self.smooth) / (
                torch.sum(skel_pred[:, 1:, ...]) + self.smooth
        )
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:, 1:, ...]) + self.smooth) / (
                torch.sum(skel_true[:, 1:, ...]) + self.smooth
        )
        cl_dice: torch.Tensor = 2.0 * (tprec * tsens) / (tprec + tsens)
        return cl_dice

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        tprec = self.cl_score(y_pred.cpu().numpy(), skeletonize(y_true.cpu().numpy()))
        tsens = self.cl_score(y_true.cpu().numpy(), skeletonize(y_pred.cpu().numpy()))
        return 2 * tprec * tsens / (tprec + tsens)
