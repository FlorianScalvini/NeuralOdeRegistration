import glob
import torchio as tio
import pytorch_lightning as pl
from dataset import  SpatioTemporalDatasetValidation, SpatioTemporalDataset
import pandas as pd
import random
import os
import torch
import json


def split_and_shuffled(lst, ratio, seed=None):
    lst_copy = lst.copy()  # avoid modifying original list
    if seed is not None:
        random.seed(seed)  # for reproducibility
    random.shuffle(lst_copy)
    split_idx = int(len(lst_copy) * ratio)
    return lst_copy[:split_idx], lst_copy[split_idx:]


class SpatioTemporalSequenceDatamoduleJSON(pl.LightningDataModule):
    def __init__(self, root_dir, json_path: str, json_path_val: str, batch_size: int, seed=42, num_workers=4, t0=0, tn=1, save_config_dir: str=""):
        super().__init__()
        self.root_dir = root_dir
        self.json_path = json_path
        self.json_path_val = json_path_val
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_subjects = None
        self.seed = seed
        self.transform = tio.Compose([

            tio.CropOrPad((224,224,224)),
            tio.Resize((128,128,128)),
            tio.RescaleIntensity(out_min_max=(0,1), percentiles=(0.05,99.5)),
        ])
        self.data_train = []
        self.data_val = []

        with open(json_path, 'r') as f:
            # Parsing the JSON file into a Python dictionary
            data = json.load(f)

        for i in range(len(data['subjects'])):
            subject = []
            for j in range(len(data['subjects'][i]['sessions'])):
                session = [
                    root_dir + data['subjects'][i]['sessions'][j]['image'],
                    root_dir + data['subjects'][i]['sessions'][j]['segmentation'],
                    data['subjects'][i]['sessions'][j]['age']
                ]
                subject.append(session)

            for j in range(len(subject)):
                subject[j][2] = (subject[j][2] - t0) / (tn - t0)
            subject.sort(key=lambda x: x[2])
            self.data_train.append(subject)

        with open(json_path_val, 'r') as f:
            # Parsing the JSON file into a Python dictionary
            data = json.load(f)
        for i in range(len(data['subjects'])):
            subject = []
            for j in range(len(data['subjects'][i]['sessions'])):
                session = [
                    root_dir + data['subjects'][i]['sessions'][j]['image'],
                    root_dir + data['subjects'][i]['sessions'][j]['segmentation'],
                    data['subjects'][i]['sessions'][j]['age']
                ]
                subject.append(session)

            for j in range(len(subject)):
                subject[j][2] = (subject[j][2] - t0) / (tn - t0)
            subject.sort(key=lambda x: x[2])
            self.data_val.append(subject)




    def train_dataloader(self) -> torch.utils.data.DataLoader:
        dataset = SpatioTemporalDataset(self.data_train, self.transform)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=1,             # ← Single worker
            shuffle=True,
            prefetch_factor=1,         # ← Only 1 batch ahead (your current setting)
            pin_memory=True,
            persistent_workers=False,
            drop_last=False,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        dataset = SpatioTemporalDatasetValidation(self.data_val, self.transform)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=1,             # ← Single worker
            shuffle=False,
            prefetch_factor=1,         # ← Only 1 batch ahead (your current setting)
            pin_memory=False,
            persistent_workers=False,
            drop_last=False,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        dataset = SpatioTemporalDatasetValidation(self.data_val, self.transform)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=1,             # ← Single worker
            shuffle=False,
            prefetch_factor=1,         # ← Only 1 batch ahead (your current setting)
            pin_memory=False,
            persistent_workers=False,
            drop_last=False,
        )


