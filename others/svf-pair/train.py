import argparse
import os
import gc
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import registration_svf
from registration_svf.registration import RegistrationModule
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from datamodule import SpatioTemporalSequenceDatamoduleJSON
from training_module import RegistrationTrainingModule
from datetime import datetime
gc.collect()
torch.cuda.empty_cache()
import registration_svf.modules.unet as unet

def main(args) -> None:
    torch.set_float32_matmul_precision('high')

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
    tensorboard_logger: pl.loggers.TensorBoardLogger = pl.loggers.TensorBoardLogger(
        save_dir=save_dir
    )
    model = unet.DyNUnet(
        in_channels=2,
        out_channels=3,
        kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        strides=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        dropout=0.1
    )
    model = RegistrationModule(model=model)


    datamodule: pl.LightningDataModule = SpatioTemporalSequenceDatamoduleJSON(
        root_dir=args.root_dir,
        json_path=args.json_path,
        json_path_val=args.json_path_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        size=args.size,
        crop=args.crop
    )

    training_module = RegistrationTrainingModule(model=model,
                                                 save_path=save_dir,
                                                 learning_rate=args.learning_rate,
                                                 lambda_sim=args.lambda_sim,
                                                 lambda_reg=args.lambda_reg,
                                                 lambda_seg=args.lambda_seg)

    trainer = pl.Trainer(max_epochs=5000, precision=32, num_sanity_val_steps=30, logger=tensorboard_logger,
                         callbacks= [ModelCheckpoint(every_n_train_steps=args.checkpoint_every_n_steps, dirpath=save_dir, save_last=True)],
                         check_val_every_n_epoch=20, gradient_clip_algorithm='norm',
                         enable_progress_bar=True)
    
    checkpoint = None
    if args.checkpoint != "":
        checkpoint = args.checkpoint
    trainer.fit(model=training_module,
                datamodule=datamodule,
                ckpt_path=checkpoint)
    print("Training finished")
    print("Saving model")
    training_module.save(save_dir)


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
        default=0.01,
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
        "--check_val_every_n_steps",
        type=int,
        default=100,
        help="Run validation every N training steps.",
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
