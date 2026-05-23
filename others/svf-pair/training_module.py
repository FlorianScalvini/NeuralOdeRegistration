import os
import gc
import torch
import monai
import torch.nn as nn
import torchio as tio
import pytorch_lightning as pl
from monai.metrics import DiceMetric
from registration_svf.losses.regularisation.jacobian import Jacobianloss, compute_jacobian_determinant_3d, compute_jacobian_determinant, jacobian_determinant_3d
from registration_svf.registration import RegistrationModule
from registration_svf.utils.grid_utils import warp, compose, displacement2grid, plt_grid
from registration_svf.losses.regularisation.magnitude import  MagnitudeLoss
from registration_svf.losses.regularisation.gradient import  Grad3d
import torch.nn.functional as F
from registration_svf.utils.utils import normalize_to_0_1
from torchvision.utils import make_grid
from torchvision.transforms import transforms
import numpy as np

class RegistrationTrainingModule(pl.LightningModule):
    """
    Registration training module for 3D image registration
    """
    def __init__(self, model : RegistrationModule, learning_rate: float= 0.001, save_path: str = "./",
                 lambda_sim=1.0, lambda_seg=0.0, lambda_reg=0.0):
        """
        Registration training module for 3D image registration
        :param model: RegistrationModule
        :param learning_rate: Learning rate for the optimizer
        :param save_path: Path to save the model
        :param lambda_sim: Loss factor - intensity image
        :param lambda_seg: Loss factor - segmentation map
        :param lambda_reg: Loss factor - segmentation map
        """

        super().__init__()
        self.reg_model = model
        self.save_path = save_path
        self.learning_rate = learning_rate
        self.seg_metrics = DiceMetric(include_background=True, reduction="none", ignore_empty=False)
        self.dice_max = 0
        self.sim_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(kernel_size=21)
        self.seg_loss = nn.MSELoss()
        self.lambda_seg = lambda_seg
        self.lambda_sim = lambda_sim
        self.lambda_reg = lambda_reg
        self.loss_reg = Grad3d('l2')
        self.table_result_data = []
        self.val_grid_images = []
        self.max_dice_score = 0
        self.automatic_optimization = False

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the registration module
        :param source: Source image
        :param target: Target image
        :return: Flow field if RegistrationModule else
        """
        return self.reg_model(torch.cat([source, target], dim=1))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_epoch_start(self) -> None:
        self.dice_max = 0
        self.reg_model.train()

    def training_step(self, batch) -> torch.Tensor:
        opt = self.optimizers()
        images, segs = batch
        shape = images[0].shape[2:]
        images = images.squeeze(0)
        shape = images.shape[2:]
        initial_seg = F.one_hot(segs[:, 0].squeeze(0).cpu().long(), num_classes=-1).permute(0, 4, 1, 2, 3).to(self.device)
        source_img = images[0:1].float()
        for idx in range(0, images.shape[0]):
            loss_errors = torch.zeros([3], device=self.device)
            target_img = images[idx:idx+1].float()
            velocity = self.forward(source_img, target_img)
            t = torch.rand(1, device=self.device)
            disp_j = self.reg_model.velocity2displacement(velocity * t)
            disp_i = self.reg_model.velocity2displacement(velocity * (t - 1.))
            if self.lambda_sim > 0:
                jw = warp(source_img, disp_j)
                iw = warp(target_img, disp_i)
                loss_errors[0] = self.lambda_sim * self.sim_loss(jw, iw)
            if self.lambda_seg > 0:
                target_label = F.one_hot(segs[:, idx].squeeze(0).cpu().long(), num_classes=initial_seg.shape[1]).permute(0, 4, 1, 2, 3).to(self.device)
                jw = warp(initial_seg.float(), disp_j)
                iw = warp(target_label.float(), disp_i)
                loss_errors[1] = self.lambda_seg * (self.seg_loss(jw, iw))
            # Gradient loss
            if self.lambda_reg > 0:
                loss_errors[2] += self.lambda_reg * (self.loss_reg(disp_j) + self.loss_reg(disp_i))

            loss = loss_errors.sum()

            self.log_dict({
                "Global loss": loss,
                "Intensity" : loss_errors[0],
                "Segmentation": loss_errors[1],
                "Regulation": loss_errors[2]
            }, prog_bar=True, on_epoch=True, sync_dist=True)
            opt.zero_grad(set_to_none=True)
            self.manual_backward(loss)
            opt.step()
            del velocity, disp_j, disp_i, loss_errors, target_img
            torch.cuda.empty_cache()

    def on_train_epoch_end(self):
        torch.save(self.reg_model.state_dict(), self.save_path + "/last_model.pth")
        gc.collect()

    def validation_step(self, batch, batch_idx) -> None:
        images, segs = batch
        shape = images[0].shape[2:]
        scale_factor = torch.tensor(shape).to(self.device).view(1, 3, 1, 1, 1) * 1.
        images = images.squeeze(0)
        shape = images.shape[2:]
        all_registered = []
        all_targets = []
        all_segs = []
        initial_seg = F.one_hot(segs[:, 0].squeeze(0).cpu().long(), num_classes=-1).permute(0, 4, 1, 2, 3)
        for idx in range(0, images.shape[0]):
            with torch.no_grad():
                velocity = self.forward(images[0:1].float(), images[idx:idx+1].float())
                df = self.reg_model.velocity2displacement(velocity)   
                warped = warp(images[0:1].float(), df)
                warped_seg = warp(initial_seg.to(self.device).float(), df)
                warped_seg = torch.argmax(warped_seg, dim=1).detach()
                pred_label = F.one_hot(warped_seg.cpu().long(), num_classes=initial_seg.shape[1]).permute(0, 4, 1, 2, 3)

                all_registered.append(
                    normalize_to_0_1(warped.squeeze())[:, :, shape[-1] // 2].detach().cpu().unsqueeze(0).repeat(3, 1, 1)
                )
                all_targets.append(
                    normalize_to_0_1(images[idx].squeeze(0))[:, :, shape[-1] // 2].detach().cpu().unsqueeze(0).repeat(3, 1,
                                                                                                                    1)
                )
                all_segs.append(
                    normalize_to_0_1(warped_seg.squeeze())[:, :, shape[-1] // 2].detach().cpu().unsqueeze(0).repeat(3, 1, 1)
                )
                xy = displacement2grid(df.cpu()).squeeze(0).detach()
                grid_img = plt_grid(xy[:, :, shape[-1] // 2, :].cpu())[0]
                to_tensor = transforms.ToTensor()
                grid_img = to_tensor(grid_img)  # (3, H, W)

                if idx != 0:
                    self.seg_metrics(pred_label, F.one_hot(segs[:, idx].squeeze(0).cpu().long(),
                                                        num_classes=initial_seg.shape[1]).permute(0, 4, 1, 2, 3).cpu())
                    det_jac = compute_jacobian_determinant_3d(df.cpu()).numpy()
                    nb_jac_neg = int(np.sum(det_jac < 0))
                    buffer = self.seg_metrics.get_buffer()
                    dice = float(buffer[-1].mean().item())
                    results = [str(batch_idx) + "_" + str(idx), grid_img, dice, nb_jac_neg]
                    self.table_result_data.append(results)

                del warped, warped_seg, df, xy, pred_label
                torch.cuda.empty_cache()


        torch.cuda.empty_cache()

        num_times = images.shape[0] 
        combined = torch.stack(all_targets + all_registered + all_segs)
        grid_visualization = make_grid(combined, nrow=num_times, padding=5, pad_value=1.0)
        self.val_grid_images.append(grid_visualization)
        del combined


    def on_validation_epoch_end(self) -> None:
        if not self.table_result_data:  # skip sanity check
            self.seg_metrics.reset()
            return

        step = self.current_epoch

        # Log temporal comparison grids
        for i, img in enumerate(self.val_grid_images):
            self.logger.experiment.add_image(
                f"Temporal_Comparison/batch_{i}",
                img,
                global_step=step
            )

        # Log grid images + scalars as a combined image panel
        grid_imgs = [row[1] for row in self.table_result_data]  # tensors (3,H,W)
        dice_vals = [row[2] for row in self.table_result_data]
        jac_vals = [row[3] for row in self.table_result_data]

        if grid_imgs:
            grid_panel = make_grid(torch.stack(grid_imgs), nrow=len(grid_imgs), padding=2, pad_value=1.0)
            self.logger.experiment.add_image("Grid/all", grid_panel, global_step=step)

        # Log per-sample scalars
        for row in self.table_result_data:
            sample_id, _, dice, nb_jac_neg = row
            self.logger.experiment.add_scalar(f"Dice/{sample_id}", dice, global_step=step)
            self.logger.experiment.add_scalar(f"JacNeg/{sample_id}", nb_jac_neg, global_step=step)

        mean_dice = float(np.mean(dice_vals))
        # Log mean dice and jac
        self.logger.experiment.add_scalar("Val/mean_dice", mean_dice, global_step=step)
        self.logger.experiment.add_scalar("Val/mean_jac_neg", float(np.mean(jac_vals)), global_step=step)

        # Reset
        self.table_result_data = []
        self.val_grid_images = []

        if self.max_dice_score < mean_dice:
            self.max_dice_score = mean_dice
            torch.save(self.reg_model.state_dict(), os.path.join(self.save_path, "model.pth"))
        torch.cuda.empty_cache()



    def on_validation_epoch_start(self) -> None:
        self.jacobian_nb_value = 0


    def save(self, path: str):
        """
        Save the model
        :param path: Path to save the model
        """
        torch.save(self.reg_model.state_dict(), os.path.join(path, "model.pth"))
