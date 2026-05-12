import os
import gc

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from registration_svf.registration import RegistrationModule
from omegaconf import DictConfig
from dataloader import SpatioTemporalSequenceDatamoduleJSON
from model import RegistrationLongitudinal
import torchio as tio
from registration_svf.modules.monotonic_mlp import MonotonicMLP
import torch.nn.functional as F
from datetime import datetime


gc.collect()
torch.cuda.empty_cache()
def main() -> None:
    torch.set_float32_matmul_precision('high')
    save_dir = "./calgary_longitudinal_subject"
    dir_name = datetime.now().strftime("%y_%d_%H_%M")
    save_dir = os.path.join(save_dir, dir_name)
    os.makedirs(save_dir, exist_ok=True)
    tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(save_dir))

    datamodule: pl.LightningDataModule = SpatioTemporalSequenceDatamoduleJSON(
        root_dir="/home/florian/Documents/Dataset/Calgary/",
        json_path="/home/florian/Documents/Dataset/Calgary/data.json",
        batch_size=1,
        num_workers=8,
        t0=6,
        tn=140,
        save_config_dir=save_dir
    )
    training_module = RegistrationLongitudinal(learning_rate=0.001, save_dir=save_dir)

    trainer = pl.Trainer(max_epochs=2000, precision=32, num_sanity_val_steps=30,
                         logger=tensorboard_logger,
                         callbacks=[ModelCheckpoint(every_n_train_steps=50, dirpath=save_dir, save_last=True)],
                         check_val_every_n_epoch=10,
                         enable_progress_bar=True)
    checkpoint = "/home/florian/PyCharmMiscProject/src/calgary_longitudinal_subject/epoch=173-step=11300.ckpt"
    if checkpoint is None:
        trainer.test()
    else:
        trainer.test(ckpt_path=checkpoint)



if __name__ == '__main__':
    main()
