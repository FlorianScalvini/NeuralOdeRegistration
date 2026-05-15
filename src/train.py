import argparse
import gc
import os
from argparse import Namespace
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from dataloader import SpatioTemporalSequenceDatamoduleJSON
from model import RegistrationLongitudinal



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
    os.makedirs(save_dir, exist_ok=True)

    # --- Logger ---
    tensorboard_logger: pl.loggers.TensorBoardLogger = pl.loggers.TensorBoardLogger(
        save_dir=save_dir
    )

    # --- Data module ---
    datamodule: pl.LightningDataModule = SpatioTemporalSequenceDatamoduleJSON(
        root_dir=args.root_dir,
        json_path=args.json_path,
        json_path_val=args.json_path_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        t0=args.t0,
        tn=args.tn,
        save_config_dir=save_dir,
    )

    # --- Model ---
    training_module: RegistrationLongitudinal = RegistrationLongitudinal(
        learning_rate=args.learning_rate,
        save_dir=save_dir,
        lambda_seg=args.lambda_seg,
        lambda_reg=args.lambda_reg,
    )

    # --- Trainer ---
    trainer: pl.Trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        num_sanity_val_steps=args.num_sanity_val_steps,
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
        default="/home/florian/Documents/Dataset/dHCP/Atlas",
        help="Root directory containing the dataset.",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="/home/florian/Documents/Dataset/dHCP/Atlas/data.json",
        help="Path to the training JSON data file.",
    )
    parser.add_argument(
        "--json_path_val",
        type=str,
        default="/home/florian/Documents/Dataset/dHCP/Atlas/data.json",
        help="Path to the validation JSON data file.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./calgary_longitudinal_subject",
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
        default=21,
        help="Start gestational age (weeks).",
    )
    parser.add_argument(
        "--tn",
        type=int,
        default=36,
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
        default=1000,
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
        "--lambda_reg",
        type=float,
        default=0.0001,
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
        default=5,
        help="Run validation every N epochs.",
    )
    parser.add_argument(
        "--checkpoint_every_n_steps",
        type=int,
        default=50,
        help="Save a checkpoint every N training steps.",
    )
    args = parser.parse_args()
    main(args=args)
