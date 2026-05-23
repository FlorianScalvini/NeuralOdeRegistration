import argparse
import gc
import os
from argparse import Namespace
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from datamodule import SpatioTemporalSequenceDatamoduleJSON
from training_module import LongitudinalTrainingModule
from longitudinal_model import LongitudinalDeformation
import registration_svf as svf
import monai


def main(args: Namespace) -> None:
    """Build all components and launch training.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments returned by :func:`parse_args`.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("high")

    # --- Output directory ---
    dir_name: str = datetime.now().strftime("%y_%d_%H_%M")
    save_dir: str = os.path.join(args.save_dir, dir_name)
    if os.path.exists(save_dir):
        # create versioned directory if the base directory already exists
        version = 1
        while os.path.exists(f"{save_dir}_v{version}"):
            version += 1
        save_dir = f"{save_dir}_v{version}"
    os.makedirs(save_dir, exist_ok=True)

    # --- Logger ---
    tensorboard_logger: pl.loggers.TensorBoardLogger = pl.loggers.TensorBoardLogger(
        save_dir=save_dir
    )

    model = svf.modules.unet.DyNUnet(
        in_channels=2,
        out_channels=3,
        kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        strides=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        dropout=0.1
    )
    model = svf.registration.RegistrationModule(model=model)
    model : LongitudinalDeformation = LongitudinalDeformation(svf_model=model, time_mode='linear', t0=args.t0, t1=args.tn)
    training_module = LongitudinalTrainingModule(model=model,save_path=args.save_dir,  
                                                 learning_rate_svf=args.learning_rate, 
                                                 learning_rate_mlp=args.learning_rate, lambda_reg=args.lambda_reg, lambda_sim=args.lambda_sim, lambda_seg=args.lambda_seg)

    datamodule: pl.LightningDataModule = SpatioTemporalSequenceDatamoduleJSON(
        root_dir=args.root_dir,
        json_path=args.json_path,
        json_path_val=args.json_path_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        size=args.size,
        crop=args.crop,
        t0=args.t0,
        tn=args.tn
    )

    # --- Trainer ---
    trainer: pl.Trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        num_sanity_val_steps=0,
        logger=tensorboard_logger,
        callbacks=[
            ModelCheckpoint(
                every_n_train_steps=args.checkpoint_every_n_steps,
                dirpath=save_dir,
                save_last=True,
            ),
            TQDMProgressBar(refresh_rate=1),
        ],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        enable_progress_bar=True,
    )

    trainer.fit(
        model=training_module,
        datamodule=datamodule,
        ckpt_path=args.checkpoint,
    )
    # trainer.test(model=training_module, datamodule=datamodule, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the longitudinal brain MRI registration model."
    )
    # --- Paths ---
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/florian/Documents/Dataset/Calgary",
        help="Root directory containing the dataset.",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="/home/florian/Documents/Dataset/Calgary/data_train.json",
        help="Path to the training JSON data file.",
    )
    parser.add_argument(
        "--json_path_val",
        type=str,
        default="/home/florian/Documents/Dataset/Calgary/data_val.json",
        help="Path to the validation JSON data file.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./dhcp_longitudinal_subject",
        help="Base directory where model checkpoints and logs are saved.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from (optional).",
    )

    # --- Data ---
    parser.add_argument(
        "--t0",
        type=int,
        default=6,
        help="Start gestational age (weeks).",
    )
    parser.add_argument(
        "--tn",
        type=int,
        default=125,
        help="End gestational age (weeks).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader worker processes.",
    )

    # --- Training ---
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=3000,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--lambda_seg",
        type=float,
        default=1.0,
        help="Weight for the segmentation loss term.",
    )
    parser.add_argument(
        "--lambda_sdf",
        type=float,
        default=0.0,
        help="Weight for the SDF loss term.",
    )
    parser.add_argument(
        "--lambda_sim",
        type=float,
        default=0.0,
        help="Weight for the similarity loss term.",
    )
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=0.05,
        help="Weight for the regularisation loss term.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        choices=[16, 32],
        help="Floating-point precision used during training.",
    )
    parser.add_argument(
        "--num_sanity_val_steps",
        type=int,
        default=30,
        help="Number of sanity validation steps before training starts.",
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=10,
        help="Run validation every N epochs.",
    )
    parser.add_argument(
        "--checkpoint_every_n_steps",
        type=int,
        default=50,
        help="Save a checkpoint every N training steps.",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=[192, 192, 192],
        nargs="+",
        help="Size of resize.",
    )
    parser.add_argument(
        "--crop",
        type=int,
        default=[192, 192, 192],
        nargs="+",
        help="Size of crop.",
    )
    args = parser.parse_args()
    main(args=args)
