import monai.metrics
import os

from pytorch_lightning.utilities.types import STEP_OUTPUT
import torchio as tio
from utils import *
from torchvision import transforms
from unet import *
from metrics.sdlogjac import SDlogDetJac
from torchvision.utils import make_grid
import pystrum.pynd.ndutils as nd
from torchvision.utils import save_image
import json
import itertools

class TimeEncoder(nn.Module):
    def __init__(self, t_dim: int = 64, dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(t_dim, dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.LayerNorm(dim))

    def forward(self, T: torch.Tensor) -> torch.Tensor:
        return self.mlp(T)


class TemporalFusion(nn.Module):
    """
    A single learned query cross-attends over N time-aware feature
    vectors to produce a fixed-size subject context vector z.

    Why no positional encoding:
        Each fᵢ was produced by TimeConditionedEncoderUnet which
        received (image ‖ age_broadcast) as a 2-channel input.
        The age is therefore encoded in the spatial features
        themselves — the query reads temporal position implicitly
        from feature content, not from an external PE signal.

    Why N=1 works without special casing:
        Softmax over a single key collapses to weight=1.0.
        The query attends fully to the only available vector.
        z becomes a clean projection of that single feature.
        No zeroed statistics, no instability, no if-branch.

    Why weights vary per subject:
        The query is a learned parameter shared across all subjects
        and all values of N.  It learns to ask "which timepoint
        tells me most about this trajectory?".  For a subject with
        rapid late atrophy the last fᵢ will be most distinctive
        and receive the highest weight.  For a stable subject the
        weights will be distributed evenly.  This emerges from
        the registration + segmentation loss — no explicit
        supervision on the attention weights is needed.

    Args:
        feat_dim : dimension of each per-image feature vector,
                   i.e. the output dim of ImageFeaturePooler
        z_dim    : output context vector dimension
        nhead    : number of attention heads in cross-attention.
                   Must divide feat_dim evenly.
        dropout  : attention dropout (0 is fine for small N)
    """

    def __init__(self,
                 feat_dim: int = 256,
                 z_dim:    int = 512,
                 nhead:    int = 4,
                 dropout:  float = 0.0):
        super().__init__()
        self.feat_dim = feat_dim

        # ── Learned query ────────────────────────────────────────
        # Shape (1, 1, feat_dim): one query vector, expanded to
        # batch size in forward().  Initialised small so early
        # training is close to uniform attention over timepoints.
        self.query = nn.Parameter(
            torch.randn(1, 1, feat_dim) * 0.02)

        # ── Cross-attention ──────────────────────────────────────
        # query  : (B, 1, feat_dim)   — the learned question
        # key    : (B, N, feat_dim)   — one token per timepoint
        # value  : (B, N, feat_dim)   — same tokens as values
        # output : (B, 1, feat_dim)   — weighted summary
        #
        # batch_first=True means tensors are (B, seq, dim)
        # which matches our layout throughout.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True)

        # ── Output projection ────────────────────────────────────
        # Projects the attended feat_dim vector to z_dim and
        # normalises — keeps z on a consistent scale regardless
        # of feat_dim choice.
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, z_dim),
            nn.LayerNorm(z_dim),
        )

    def forward(self,
                features: torch.Tensor,
                return_weights: bool = False
                ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        B = features.shape[0]

        # Expand the single learned query to the full batch
        # (1, 1, feat_dim) → (B, 1, feat_dim)
        q = self.query.expand(B, -1, -1)

        # Cross-attention
        # need_weights=True so we can optionally inspect what the
        # query attended to.  average_attn_weights=False keeps
        # per-head weights if nhead > 1 — we average manually below
        # so the returned shape is always (B, 1, N).
        attended, attn_weights = self.cross_attn(
            query=q,
            key=features,
            value=features,
            need_weights=True,
            average_attn_weights=True)
        # attended    : (B, 1, feat_dim)
        # attn_weights: (B, 1, N)   — softmax weights over timepoints

        # Squeeze the singleton sequence dim and project
        z = self.proj(attended.squeeze(1))   # (B, z_dim)

        if return_weights:
            return z, attn_weights           # (B, z_dim), (B, 1, N)
        return z



class ImageFeaturePooler(nn.Module):
    """
    Spatial average pooling + linear projection.

    Input  : (B, in_channels, H, W, D)   CNN bottleneck feature map
    Output : (B, feat_dim)               one vector per image

    AdaptiveAvgPool3d(1) averages every spatial position into a
    single value per channel — no parameters, no spatial bias.
    The Linear + LayerNorm after it re-projects to feat_dim and
    stabilises the scale fed into cross-attention.

    Args:
        in_channels : bottleneck channels from EncoderUnet (256)
        feat_dim    : output dimension, must match TemporalFusion
    """

    def __init__(self, in_channels: int = 256, feat_dim: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, feat_dim),
            nn.LayerNorm(feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, H, W, D) → (B, feat_dim)"""
        return self.proj(self.pool(x))

class ContextEncoder(nn.Module):
    def __init__(self, t_dim=16, ):
        super().__init__()
        self.encoder_img = EncoderUnet(in_channels=1, channels=[16, 32, 64, 128, 256], t_dim=16)
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(t_dim, max_periods=100),
                nn.Linear(t_dim, t_dim, bias=True),
                nn.SiLU()
        )
        self.img_pool = ImageFeaturePooler()
        self.temp_fusion =TemporalFusion(z_dim=512, nhead=4, dropout=0.0)

    def forward(self, imgs : torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        feat_list = []
        n = imgs.shape[0]
        for i in range(n):
            t_enc = self.time_mlp(t[i].unsqueeze(0))
            feat_list.append(self.img_pool(self.encoder_img(imgs[i].unsqueeze(0), t_enc)[-1]))
        features = torch.stack(feat_list, dim=1)
        return self.temp_fusion(features)


class ODEFunction(nn.Module):
    def __init__(self, vnet, imageA, imageB, ageA, ageB):
        super().__init__()
        self.vnet = vnet
        self.imageA = imageA
        self.imageB = imageB
        self.ageA = ageA
        self.ageB = ageB


    def forward(self, t, phi_t):
        v = self.vnet(t, phi_t, self.imageA, self.imageB, self.ageA, self.ageB)
        return v


import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal Position Embeddings for Time.
    """

    def __init__(self,
                 embed_dim: int,
                 max_periods: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_periods = max_periods

    def forward(self, x: torch.Tensor):
        '''
        Args:
            x (torch.Tensor):  time indices, float, shape (B,)

        Returns:
            embeddings (torch.Tensor): (B, embed_dim)
        '''
        indices = torch.arange(0, self.embed_dim // 2).float().to(x.device)  # (embed_dim//2,)
        indices = torch.pow(self.max_periods, -2 * indices / self.embed_dim)
        embeddings = torch.einsum('b,d->bd', x, indices)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)  # (B, embed_dim)
        return embeddings


class VelocityNet(nn.Module):
    def __init__(self,reg_head_chan=16,  z_dim=512):
        super().__init__()
        self.shape = [192,192,192]

        self.grid = generate_grid3d_tensor(self.shape).cuda()
        t_dim_enc = 16
        t_dim = 48
        #self.model = TransMorphCascadeAdFullRes(grid)
        self.encoder = EncoderUnet(in_channels=3, channels=[16, 32, 64, 128, 256], t_dim=t_dim)
        self.decoder_0 = UnetUpBlock(in_channels=256, out_channels=128, kernel_size=3, t_dim=t_dim)
        self.decoder_1 = UnetUpBlock(in_channels=128, out_channels=64, kernel_size=3, t_dim=t_dim)
        self.decoder_2 = UnetUpBlock(in_channels=64, out_channels=32, kernel_size=3, t_dim=t_dim)
        self.decoder_3 = UnetUpBlock(in_channels=32, out_channels=16, kernel_size=3, t_dim=t_dim)
        self.temp_enc = SinusoidalPositionEmbeddings(t_dim_enc, max_periods=100)
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim_enc * 3, t_dim, bias=True),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim, bias=True),
        )

        '''
        self.film_0 = FiLM(z_dim=z_dim, n_chan=128)
        self.film_1 = FiLM(z_dim=z_dim, n_chan=64)
        self.film_2 = FiLM(z_dim=z_dim, n_chan=32)
        '''
        self.reg_head = nn.Sequential(
            nn.Conv3d(reg_head_chan, reg_head_chan, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(reg_head_chan, reg_head_chan, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(reg_head_chan, 3, kernel_size=3, padding=1),
        )


    def forward(self, t, phi_t, image_A, image_B, ageA, ageB):
        with torch.no_grad():
            df = phi_t - self.grid
            warped = warp(image_A, df)
            input = torch.cat([image_A, warped, image_B], dim=1)
            B = phi_t.shape[0]
        if t.dim() == 0:
            t = t.expand(B)
        if ageA.dim() == 0:
            ageA = ageA.expand(B)
        if ageB.dim() == 0:
            ageB = ageB.expand(B)

        t = self.temp_enc((t - ageA) / ageB)
        ageB = self.temp_enc(ageB)
        ageA = self.temp_enc(ageA)
        t_all = torch.cat([ageA, t, ageB], dim=1)
        t_all = self.time_mlp(t_all)

        feat_maps = self.encoder(input, t_all)
        v = self.decoder_0(feat_maps[4], t_all, feat_maps[3])
        v = self.decoder_1(v, t_all, feat_maps[2])
        v = self.decoder_2(v, t_all, feat_maps[1])
        v = self.decoder_3(v, t_all, feat_maps[0])
        v = self.reg_head(v)
        return v



class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=False,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        nm = nn.InstanceNorm3d(out_channels)
        super(Conv3dReLU, self).__init__(conv, nm, relu)




class LongitudinalODERegistration(nn.Module):
    def __init__(self, in_channels=1, embed_dim=16,
                 depths=(1,1,2,2), num_heads=(2,2,4,8),
                 window_size=(7,7,7), patch_size=4,
                 t_dim=16, vel_dim=256, vel_heads=8,
                 drop_path_rate=0.2):
        super().__init__()
        '''
        self.encoderImage = Encoder(
            in_channels=in_channels, 
            embed_dim=embed_dim,
            depths=depths, 
            num_heads=num_heads,
            window_size=window_size,
            patch_size=patch_size,
            t_dim=t_dim, 
            drop_path_rate=drop_path_rate)
        '''
        self.velocity_net = VelocityNet()

        self.ctx_imgs = ContextEncoder()

    def forward(self, imageA: torch.Tensor, imageB: torch.Tensor, ageB: torch.Tensor, ages,
                grid: torch.Tensor) -> torch.Tensor:
        """
        image : (B, 1, H, W, D)
        ages  : (N,)  sorted integration times  ages[0] = t₀
        grid  : (B, 3, D, H, W)  identity grid in [-1, 1]

        returns: phi_all  (N, B, 3, D, H, W)  deformation field at each age
        """

        # ── 1. encode source image once ───────────────────────────
                    # (B, 1)
        #z = self.ctx_imgs(image, ages)
        # ── 5. ODE integration  dφ/dT = v(T) ─────────────────────
        ode_func = ODEFunction(self.velocity_net, imageA, imageB, ages[0], ageB)
        phi_traj = odeint(
            ode_func,
            grid,
            ages,
            method='rk4')
        return phi_traj





class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, activation=True):
        super().__init__()
        self.conv = sn(nn.Conv3d(in_ch, out_ch, 4, stride, 1))
        self.activation = activation
        self.norm = nn.InstanceNorm3d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            x = self.act(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.down1 = ConvBlock(1, channels)
        self.down2 = ConvBlock(channels, channels * 2)
        self.down3 = ConvBlock(channels * 2, channels * 4)
        self.down4 = ConvBlock(channels * 4, channels * 8, stride=1)

        # Age embeddings at different scales (use actual channel counts)
        self.age_emb1 = nn.Linear(1, channels)
        self.age_emb2 = nn.Linear(1, channels * 2)
        self.age_emb3 = nn.Linear(1, channels * 4)
        self.age_emb4 = nn.Linear(1, channels * 8)

        self.final = ConvBlock(channels * 8, 1, stride=1, activation=False)

    def forward(self, x, age):
        # Inject age at each scale
        h = self.down1(x)  # Output: [B, channels, ...]
        h = h + self.age_emb1(age).view(-1, 32, 1, 1, 1)  # Use channels value

        h = self.down2(h)  # Output: [B, channels*2, ...]
        h = h + self.age_emb2(age).view(-1, 64, 1, 1, 1)  # channels * 2

        h = self.down3(h)  # Output: [B, channels*4, ...]
        h = h + self.age_emb3(age).view(-1, 128, 1, 1, 1)  # channels * 4

        h = self.down4(h)  # Output: [B, channels*8, ...]
        h = h + self.age_emb4(age).view(-1, 256, 1, 1, 1)  # channels * 8

        return self.final(h)


class RegistrationLongitudinal(pl.LightningModule):
    def __init__(self, learning_rate=0.01, save_dir="", lambda_seg=1, lambda_reg=0.001, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.discriminator = Discriminator(channels=32)
        self.registration = LongitudinalODERegistration()
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.loss_seg = monai.losses.DiceCELoss()
        self.loss = monai.losses.LocalNormalizedCrossCorrelationLoss(kernel_size=21)
        self.discrimation_loss = nn.BCEWithLogitsLoss()
        self.loss_reg = monai.losses.BendingEnergyLoss(True)
        self.max_dice_score = 0
        self.automatic_optimization = False
        self.discriminator_step = False
        self.lambda_d = 1
        self.lambda_reg = lambda_reg
        self.lambda_seg = lambda_seg
        self.seg_metrics = monai.metrics.DiceMetric()
        self.table_result_data = []
        self.val_grid_images = []
        self.test_result = {"mDice": 0.0, "subjects": []}


    def configure_optimizers(self):
        opt_G = torch.optim.Adam(self.registration.parameters(), lr=self.learning_rate)
        lr_scheduler_G = torch.optim.lr_scheduler.ExponentialLR(opt_G, gamma=0.999)
        return [opt_G, opt_G], [lr_scheduler_G, lr_scheduler_G]


    def forward(self, source, target, age_target, ages, grid):
        return self.registration(source, target, age_target, ages, grid)

    def training_step(self, batch, batch_idx):
        opt_G, opt_D = self.optimizers()

        images, segs, ages, is_false = batch
        shape = images[0].shape[2:]
        scale_factor = torch.tensor(shape).to(self.device).view(1, 3, 1, 1, 1) * 1.
        grid = generate_grid3d_tensor(shape).unsqueeze(0).to(self.device)

        images = images.squeeze(0)
        ages = ages.squeeze(0).to(self.device)






        count = 0
        count_real = 0
        pairIndexes = itertools.combinations(range(images.shape[0]), 2)
        for pairIdx in pairIndexes:
            loss_sim = torch.zeros(1, device=self.device)
            loss_seg = torch.zeros(1, device=self.device)
            loss_reg = torch.zeros(1, device=self.device)
            loss_d = torch.zeros(1, device=self.device)
            initial_img = images[pairIdx[0]:pairIdx[0]+1].float()
            target_img = images[pairIdx[1]:pairIdx[1]+1].float()
            initial_seg = F.one_hot(segs[:, pairIdx[0]].squeeze(0).cpu().long(), num_classes=-1).permute(0, 4, 1, 2, 3)
            all_phi = self(initial_img, target_img, ages[pairIdx[0]], ages, grid)
            grid_voxel = (grid + 1.) / 2. * scale_factor
            all_phi_v = (all_phi + 1.) / 2. * scale_factor
            for idx in range(pairIdx[0]+1, images.shape[0]):
                phi = all_phi_v[idx]
                df = phi - grid_voxel
                warped = warp(initial_img, df)
                warped_seg = warp(initial_seg.float().to(self.device), df)
                loss_sim += self.loss(warped, images[idx:idx + 1].float())
                loss_seg += self.loss_seg(warped_seg, F.one_hot(segs[:, idx].squeeze(0).cpu().long(), num_classes=initial_seg.shape[1]).permute(0, 4, 1, 2, 3).float().to(self.device))
                count_real += 1
                loss_reg += self.loss_reg(df)
                del warped, phi, df
                count += 1
            loss_seg /= max(count, 1)
            loss_reg /= max(count, 1)
            loss_sim /= max(count, 1)
            loss_d /= max(count, 1)
            loss =  loss_sim + self.lambda_seg * loss_seg  + self.lambda_reg * loss_reg + self.lambda_d * loss_d
            opt_G.zero_grad()
            #self.clip_gradients(opt_G, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            self.manual_backward(loss)
            opt_G.step()

            self.log_dict({
                'loss_G': loss.item(),
                'loss_sim': loss_sim.item(),
                'loss_seg': (self.lambda_seg * loss_seg).item(),
                'loss_reg': (self.lambda_reg * loss_reg).item(),
            }, on_step=True, on_epoch=True, prog_bar=True)

        # ── critical: free the ODE trajectory ──
        del all_phi, all_phi_v, grid_voxel, loss, loss_sim, loss_reg, loss_d
        # ── always flush at end of step ──
        torch.cuda.empty_cache()

    def on_train_epoch_end(self) -> None:
        self.discriminator_step = False
        torch.cuda.empty_cache()  # ← add this
        torch.save(self.discriminator.state_dict(), os.path.join(self.save_dir, "last_discriminator.pt"))
        torch.save(self.registration.state_dict(), os.path.join(self.save_dir, "last_registration.pt"))

    def on_validation_epoch_start(self) -> None:
        self.val_grid_images = []
        self.table_result_data = []

    def validation_step(self, batch, batch_idx):
        images, segs, ages = batch
        shape = images[0].shape[2:]
        scale_factor = torch.tensor(shape).to(self.device).view(1, 3, 1, 1, 1) * 1.
        grid = generate_grid3d_tensor(shape).unsqueeze(0).to(self.device)
        images = images.squeeze(0)
        ages = ages.squeeze(0).to(self.device)
        shape = images.shape[2:]
        initial_img = images[0:1].float()
        target_img = images[-1:].float()
        with torch.no_grad():
            all_phi = self(initial_img, target_img, ages[-1], ages, grid)
        grid_voxel = (grid + 1.) / 2. * scale_factor
        all_registered = []
        all_targets = []
        all_segs = []

        initial_seg = F.one_hot(segs[:, 0].squeeze(0).cpu().long(), num_classes=-1).permute(0, 4, 1, 2, 3)
        for idx in range(0, images.shape[0]):
            phi = all_phi[idx]
            phi = (phi + 1.) / 2. * scale_factor
            df = phi - grid_voxel
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
                det_jac = Get_Ja(df.cpu()).numpy()
                nb_jac_neg = int(np.sum(det_jac < 0))
                buffer = self.seg_metrics.get_buffer()
                dice = float(buffer[-1].mean().item())
                results = [str(batch_idx) + "_" + str(idx), grid_img, dice, nb_jac_neg]
                self.table_result_data.append(results)

            del warped, warped_seg, phi, xy, pred_label
            torch.cuda.empty_cache()

        del all_phi, df
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
            torch.save(self.discriminator.state_dict(), os.path.join(self.save_dir, "best_discriminator.pt"))
            torch.save(self.registration.state_dict(), os.path.join(self.save_dir, "best_registration.pt"))

        torch.cuda.empty_cache()

    def on_test_start(self) -> None:
        self.seg_metrics.reset()

    def test_step(self, batch, batch_idx):
        images, segs, ages = batch
        shape = images[0].shape[2:]
        scale_factor = torch.tensor(shape).to(self.device).view(1, 3, 1, 1, 1) * 1.
        grid = generate_grid3d_tensor(shape).unsqueeze(0).to(self.device)
        images = images.squeeze(0)
        ages = ages.squeeze(0).to(self.device)
        shape = images.shape[2:]
        with torch.no_grad():
            all_phi = self(images[0:1].float(), ages, grid)
        grid_voxel = (grid + 1.) / 2. * scale_factor
        all_registered = []
        all_targets = []
        all_segs = []
        result_batch = {"ID": batch_idx, "sessions": []}
        initial_seg = F.one_hot(segs[:, 0].squeeze(0).cpu().long(), num_classes=-1).permute(0, 4, 1, 2, 3)
        for idx in range(images.shape[0]-1, images.shape[0]):
            print(idx)
            phi = all_phi[idx]
            phi = (phi + 1.) / 2. * scale_factor
            df = phi - grid_voxel
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
            if idx != 0:
                self.seg_metrics(pred_label, F.one_hot(segs[:, idx].squeeze(0).cpu().long(),
                                                       num_classes=initial_seg.shape[1]).permute(0, 4, 1, 2, 3).cpu())
                det_jac = Get_Ja(df.cpu()).numpy()
                nb_jac_neg = 0
                buffer = self.seg_metrics.get_buffer()
                dice = float(buffer[-1].mean().item())
                result_batch["sessions"].append({
                    "ID": batch_idx,
                    "mdice": dice,
                    "dice": buffer[-1].cpu().numpy().tolist(),
                    "nb_jac_neg": nb_jac_neg,
                })

            tio.LabelMap(tensor=warped_seg.cpu()).save(os.path.join(self.save_dir, f"seg_{batch_idx}_{idx}.nii.gz"))
            del warped, warped_seg, phi, pred_label
            torch.cuda.empty_cache()
        del all_phi, df
        torch.cuda.empty_cache()
        self.test_result['subjects'].append(result_batch)
        num_times = images.shape[0]
        combined = torch.stack(all_targets + all_registered + all_segs)
        grid_visualization = make_grid(combined, nrow=num_times, padding=5, pad_value=1.0)
        save_image(grid_visualization, os.path.join(self.save_dir, f"grid_result_{batch_idx}.png"))
        del combined

    def on_test_end(self) -> None:
        self.test_result["mDice"] = self.seg_metrics.aggregate().item()
        with open(os.path.join(self.save_dir, "result.json"), "w") as f:
            json.dump(self.test_result, f)
        torch.cuda.empty_cache()



def wrong_age(age: torch.Tensor, gap: float) -> torch.Tensor:
    """
    Sample a wrong age in [0, 1] at least `gap` away from `age`.
    Uses uniform continuous sampling — no integers.
    """
    a = age.item()

    low_end_available  = (a - gap) > 0.
    high_end_available = (a + gap) < 1.

    if not low_end_available and not high_end_available:
        raise ValueError(
            f"No valid wrong age: age={a:.3f}, gap={gap}, range=[0, 1]")

    if not low_end_available:
        # only high end: sample from [a+gap, 1]
        wrong = torch.rand(1, device=age.device) * (1. - (a + gap)) + (a + gap)

    elif not high_end_available:
        # only low end: sample from [0, a-gap]
        wrong = torch.rand(1, device=age.device) * (a - gap)

    else:
        # both ends available — pick a side randomly
        if torch.rand(1).item() > 0.5:
            # low end: [0, a-gap]
            wrong = torch.rand(1, device=age.device) * (a - gap)
        else:
            # high end: [a+gap, 1]
            wrong = torch.rand(1, device=age.device) * (1. - (a + gap)) + (a + gap)

    return wrong.view_as(age).to(age.dtype)

def Get_Ja(displacement):
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3
    '''
    displacement = displacement.squeeze().permute(1, 2, 3, 0)
    # check inputs
    volshape = displacement.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'
    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))
    # compute gradients
    J = torch.gradient(displacement + torch.Tensor(grid).to(displacement.device))

    dx = J[0]
    dy = J[1]
    dz = J[2]

    # compute jacobian components
    Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
    Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
    Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

    return Jdet0 - Jdet1 + Jdet2