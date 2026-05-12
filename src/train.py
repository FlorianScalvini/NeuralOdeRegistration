import os
import gc

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloader import SpatioTemporalSequenceDatamoduleJSON
from model import RegistrationLongitudinal
import torchio as tio
import torch.nn.functional as F
from datetime import datetime
from pytorch_lightning.callbacks import TQDMProgressBar


lambda_seg = 1
lambda_reg = 0.0001
root_dir = "/home/florian/Documents/Dataset/Calgary/"
json_path = "/home/florian/Documents/Dataset/Calgary/data_train.json"
json_path_val = "/home/florian/Documents/Dataset/Calgary/data_val.json"

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
        root_dir=root_dir,
        json_path=json_path,
        json_path_val=json_path_val,
        batch_size=1,
        num_workers=8,
        t0=6,
        tn=140,
        save_config_dir=save_dir
    )
    training_module = RegistrationLongitudinal(learning_rate=0.001, save_dir=save_dir, lambda_seg=lambda_seg, lambda_reg=lambda_reg)
    trainer = pl.Trainer(max_epochs=1000, precision=32, num_sanity_val_steps=30,
                         logger=tensorboard_logger,
                         callbacks=[ModelCheckpoint(every_n_train_steps=50, dirpath=save_dir, save_last=True), TQDMProgressBar(refresh_rate=1)],
                         check_val_every_n_epoch=5,
                         enable_progress_bar=True)
    checkpoint = None
    trainer.fit(model=training_module, datamodule=datamodule, ckpt_path=checkpoint)
    #trainer.test(model=training_module, datamodule=datamodule, ckpt_path=checkpoint)


if __name__ == '__main__':
    main()
