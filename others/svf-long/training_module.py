import os

import monai.losses
import torch
import pytorch_lightning as pl
import torchio as tio
import matplotlib.pyplot as plt
from torch import Tensor
from monai.metrics import DiceMetric
from torchvision.transforms.functional import rotate
from registration_svf.losses.regularisation.jacobian import compute_jacobian_determinant_3d
from registration_svf.utils.grid_utils import compose, warp, displacement2grid, plt_grid
from registration_svf.utils.utils import normalize_to_0_1
from longitudinal_model import LongitudinalDeformation
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from torchvision.utils import make_grid
from torchvision.utils import save_image

class LongitudinalTrainingModule(pl.LightningModule):
    """LightningModule for 4D Longitudinal Deformation Estimation"""

    def __init__(
            self,
            model: LongitudinalDeformation,
            learning_rate_svf : float = 1e-3,
            learning_rate_mlp: float = 1e-3,
            save_path: str = "./",
            num_inter_by_epoch: int = 1,
            lambda_reg: float = 0.05,
            lambda_seg: float = 0.05,
            lambda_sim: float = 0.05
    ):
        super().__init__()
        self.model = model
        self.lambda_reg = lambda_reg
        self.lambda_seg = lambda_seg
        self.lambda_sim = lambda_sim
        self.learning_rate_svf = learning_rate_svf
        self.learning_rate_mlp = learning_rate_mlp
        self.save_path = save_path
        self.num_inter_by_epoch = num_inter_by_epoch
        self.seg_metrics = DiceMetric(include_background=True, reduction="sum", ignore_empty=False)
        self.dice_max = 0.0
        self.automatic_optimization = False
        self.loss_seg = torch.nn.MSELoss(reduction='mean')
        self.loss_sim = monai.losses.LocalNormalizedCrossCorrelationLoss(kernel_size=21)
        self.table_result_data = []
        self.val_grid_images = []
        self.max_dice_score = 0

    def forward(self, source: Tensor, target: Tensor):
        return self.model((source, target))

    def configure_optimizers(self):
        opt_svf = torch.optim.Adam(self.model.svf_model.parameters(), lr=self.learning_rate_svf)
        if self.model.time_mode == 'mlp':
            opt_mlp = torch.optim.Adam(self.model.mlp_model.parameters(), lr=self.learning_rate_mlp)
        else:
            opt_mlp = opt_svf
        return [opt_svf, opt_mlp]

    def on_train_epoch_start(self):
        self.model.train()

    def on_validation_epoch_start(self):
        self.model.eval()

    def train_mlp(self, mlp_opt, batch):
        one = torch.tensor(1.0, device=self.device)
        seg_loss = torch.zeros((), device=self.device)
        images, segs, ages, is_false, sdf = batch
        images = images.squeeze(0)
        sdf = sdf.squeeze(0)
        ages = ages.squeeze(0).to(self.device)
        shape = images[0].shape[2:]
        with torch.no_grad():
            velocity = self.model.forward(torch.concat([images[0:1], images[-1:]], dim=1)).detach()
        for idx in range(1, images.shape[0]):
            t = torch.tensor([ages[idx]], device=self.device)
            with torch.no_grad():
                time = self.model.encode_time(t).detach()
            disp_j = self.model.svf_model.velocity2displacement(velocity * time).to(device=self.device)  # φ_time
            disp_i = self.model.svf_model.velocity2displacement(velocity * (time - one)).to(device=self.device)  # φ_{time-1}
            if self.lambda_sim > 0:
                jw = warp(images[0:1].to(device=self.device), disp_j)  # t0 → time
                iw = warp(images[-1:].to(device=self.device), disp_i)  # t1 → time
                int_loss += self.loss_sim(jw, images[idx:idx+1]) + self.loss_sim(iw, images[idx:idx+1])
            if self.lambda_seg > 0:
                initial_seg = F.one_hot(segs[:, 0].squeeze(0).cpu().long(), num_classes=-1).permute(0, 4, 1, 2, 3).to(device=self.device)
                last_seg = F.one_hot(segs[:, -1].squeeze(0).cpu().long(), num_classes=-1).permute(0, 4, 1, 2, 3).to(device=self.device)
                seg_idx = F.one_hot(segs[:, idx].squeeze(0).cpu().long(), num_classes=-1).permute(0, 4, 1, 2, 3).to(device=self.device)
                jw_lab = warp(initial_seg.to(device=self.device), disp_j)
                iw_lab = warp(last_seg.to(device=self.device), disp_i)
                seg_loss += self.loss_seg(jw_lab, seg_idx) + self.loss_seg(iw_lab, seg_idx)
        loss = self.lambda_seg * seg_loss
        mlp_opt.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        mlp_opt.step()
        self.log_dict(
            {
                "Loss MLP": loss,
            },
            prog_bar=True,
        )

    def train_svf(self, svf_opt, batch):
        one = torch.tensor(1.0, device=self.device)
        int_loss = torch.zeros((), device=self.device)
        seg_loss = torch.zeros((), device=self.device)
        reg_loss = torch.zeros((), device=self.device)

        images, segs, ages, is_false, sdf = batch
        images = images.squeeze(0)
        sdf = sdf.squeeze(0)
        ages = ages.squeeze(0).to(self.device)
        shape = images[0].shape[2:]
 
        velocity = self.model.forward(torch.concat([images[0:1], images[-1:]], dim=1))  # shape: (B,C,*,*,*)
        flow_v = self.model.svf_model.velocity2displacement(velocity)  # φ_{+1}
        flow__v = self.model.svf_model.velocity2displacement(-velocity)  # φ_{-1}
        for idx in range(1, images.shape[0]):
            print(idx)
            t = torch.tensor([ages[idx]], device=self.device)
            with torch.no_grad():
                time = self.model.encode_time(t).detach()
            disp_j = self.model.svf_model.velocity2displacement(velocity * time).to(device=self.device)  # φ_time
            disp_i = self.model.svf_model.velocity2displacement(velocity * (time - one)).to(device=self.device)  # φ_{time-1}
            if self.lambda_sim > 0:
                jw = warp(images[0:1].to(device=self.device), disp_j)  # t0 → time
                iw = warp(images[-1:].to(device=self.device), disp_i)  # t1 → time
                int_loss += self.loss_sim(jw, images[idx:idx+1]) + self.loss_sim(iw, images[idx:idx+1])
            if self.lambda_seg > 0:
                initial_seg = F.one_hot(segs[:, 0].squeeze(0).cpu().long(), num_classes=-1).permute(0, 4, 1, 2, 3).to(device=self.device)
                last_seg = F.one_hot(segs[:, -1].squeeze(0).cpu().long(), num_classes=-1).permute(0, 4, 1, 2, 3).to(device=self.device)
                seg_idx = F.one_hot(segs[:, idx].squeeze(0).cpu().long(), num_classes=-1).permute(0, 4, 1, 2, 3).to(device=self.device)
                jw_lab = warp(initial_seg.float(), disp_j)
                iw_lab = warp(last_seg.float(), disp_i)
                seg_loss += self.loss_seg(jw_lab, seg_idx) + self.loss_seg(iw_lab, seg_idx)
            flow_vi = compose(flow_v, disp_i)
            flow_vj = compose(flow__v, disp_j)
            grid_vi = displacement2grid(flow_vi)
            grid_vj = displacement2grid(flow_vj)
            reg_loss += torch.mean((grid_vi - grid_vj) ** 2)
        svf_opt.zero_grad(set_to_none=True)
        loss = self.lambda_sim * int_loss + self.lambda_seg * seg_loss + self.lambda_reg * reg_loss
        self.manual_backward(loss)
        svf_opt.step()
        self.log_dict(
            {
                "Loss Global": loss,
                "Loss SVF-Int": int_loss,
                "Loss SVF-Seg": seg_loss,
                "Loss Reg": reg_loss,
            },
            prog_bar=True,
        )

    def training_step(self, x, batch_idx):
        svf_opt, mlp_opt = self.optimizers()
        self.train_svf(svf_opt, x)
        # Start the MLP training after few epoch to stabilize the training
        if self.model.time_mode == "time":
            if self.current_epoch % 20 == 0:
                for _ in range(0, 10): # Process the MLP training over 10 epochs every 20 epochs
                    self.train_mlp(mlp_opt, x)


    def on_train_epoch_end(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "last_model.pth"))


    def validation_step(self, batch, batch_idx):
        images, segs, ages = batch
        shape = images[0].shape[2:]
        scale_factor = torch.tensor(shape).to(self.device).view(1, 3, 1, 1, 1) * 1.
        images = images.squeeze(0)
        ages = ages.squeeze(0).to(self.device)
        shape = images.shape[2:]
        initial_img = images[0:1].float()
        target_img = images[-1:].float()
        with torch.no_grad():
            velocity = self.model.forward(torch.concat([images[0:1], images[-1:]], dim=1)).detach()  # shape: (B,C,*,*,*)
            all_registered = []
            all_targets = []
            all_segs = []
            initial_seg = F.one_hot(segs[:, 0].squeeze(0).cpu().long(), num_classes=-1).permute(0, 4, 1, 2, 3)
            for idx in range(1, images.shape[0]):
                time = self.model.encode_time(torch.tensor([ages[idx]], device=self.device)).detach()
                df = self.model.svf_model.velocity2displacement(velocity * time)
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
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, "model.pth"))
        torch.cuda.empty_cache()

    def save(self, path: str):
        """Saves the model state dicts to disk."""
        torch.save(self.model.state_dict(), os.path.join(path, "model.pth"))

