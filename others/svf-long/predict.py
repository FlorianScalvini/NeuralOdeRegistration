#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path
import argparse
import torch
import torchio as tio
import pytorch_lightning as pl
from monai.metrics import DiceMetric
from registration_svf.utils.grid_utils import warp
from registration_svf.registration import RegistrationModule
from longitudinal_model import LongitudinalDeformation
from registration_svf.modules.unet import DyNUnet
from datetime import datetime
import pandas as pd
import yaml

# Create a new directory recursively if it does not exist
def create_directory(directory):
    '''
        Create a new directory recursively if it does not exist
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Created directory: {directory}')
    return directory

def format_number(n, max_n):
    """
    Format a number with leading zeros based on the maximum number.

    Args:
        n:
        max_n:

    Returns:

    """
    max_digits = len(str(max_n))
    return str(n).zfill(max_digits)

def test(args):
    with open(args.dataset_yaml, "r") as f:
        config = yaml.safe_load(f)
    rsize = config['rsize']
    csize = config['csize']
    t0 = config['t0']
    t1 = config['t1']
    num_classes = config['num_classes']
    csv_path = config['csv_path']
    name = config['name']
    date_format = config['date_format']
    result_save_path = f"./result/{name}/{args.time_mode}/"
    result_save_path_images = os.path.join(result_save_path, "images")
    result_save_path_flows = os.path.join(result_save_path, "flows")
    result_save_path_seg = os.path.join(result_save_path, "parcellations")
    os.makedirs(result_save_path_images, exist_ok=True)
    os.makedirs(result_save_path_seg, exist_ok=True)
    os.makedirs(result_save_path_flows, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    ## Config Dataset / Dataloader
    subjects_list = []
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        subject = tio.Subject(
            image=tio.ScalarImage(row['image']),
            label=tio.LabelMap(row['label']),
            age=str(row['age'])
        )
        subjects_list.append(subject)

    subjects_dataset = tio.SubjectsDataset(subjects_list, transform=None)

    transforms_input = tio.Compose([
            tio.CropOrPad(csize),
            tio.Resize(rsize),
            tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5), masking_method='label'),
            tio.transforms.Clamp(out_min=0, out_max=1, include=['image'])
        ])

    transforms_without_norm = tio.Compose([
        tio.CropOrPad(target_shape=csize)
    ])

    reverse_transform = tio.Compose([
        tio.CropOrPad(target_shape=subjects_dataset[0]["image"][tio.DATA].shape[1:], padding_mode='reflect')
    ])

    source_subject = subjects_dataset[t0]
    target_subject = subjects_dataset[t1]

    source_input = transforms_input(source_subject)["image"][tio.DATA].unsqueeze(0).to(device)
    target_input = transforms_input(target_subject)["image"][tio.DATA].unsqueeze(0).to(device)
    input = torch.cat([source_input, target_input], dim=1)

    reference_date_t0 = datetime.strptime(source_subject['age'], '%Y/%m/%d') if date_format else int(source_subject['age'])
    reference_date_t1 = datetime.strptime(target_subject['age'], '%Y/%m/%d') if date_format else int(target_subject['age'])


    model = RegistrationModule(
        model=DyNUnet(
        in_channels=2,
        out_channels=3,
        kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        strides=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]).eval().to(device),
        int_steps=9).eval().to(device)
    # Load models
    if args.time_mode == 'svf_int':
        model.load_state_dict(torch.load(args.load), strict=True)
    model = LongitudinalDeformation(
        time_mode=args.time_mode, t0=t0, t1=t1, reg_model=model)
    if args.time_mode == 'svf_mlp' or args.time_mode == 'svf_lin':
        model.load_state_dict(torch.load(args.load), strict=True)
    model.eval().to(device)
    with torch.no_grad():
        velocity = model.forward(input)
        source_t0 = transforms_without_norm(source_subject)
        source_image = source_t0["image"][tio.DATA].unsqueeze(0).to(device)
        if source_t0['label'].data.shape[0] == 1:
            source_t0 = tio.OneHot()(source_t0)
        source_label = source_t0["label"][tio.DATA].unsqueeze(0).to(device)
        for i in range(len(subjects_dataset)):
            subject = subjects_dataset[i]
            age = datetime.strptime(subject['age'], '%Y/%m/%d') if date_format else int(subject['age'])
            age = (age - reference_date_t0) / (reference_date_t1 - reference_date_t0)
            timed_velocity = model.encode_time(torch.Tensor([age]).to(device)) * velocity
            forward_flow = model.reg_model.velocity2displacement(timed_velocity)
            warped_source_image = warp(source_image.float(), forward_flow)
            warped_source_label = torch.argmax(warp(source_label.float(), forward_flow), dim=1).unsqueeze(0)
            j = format_number(i, len(subjects_dataset))
            if subject["label"][tio.DATA].shape[0] == 1:
                target = tio.LabelMap(tensor=subject["label"][tio.DATA].to(device),
                                     affine=subject['label'].affine)
            else:
                target = tio.LabelMap(tensor=torch.argmax(subject["label"][tio.DATA].unsqueeze(0).to(device), dim=1), affine=subject['label'].affine)

            warped_subject = tio.Subject(
                image=tio.ScalarImage(tensor=warped_source_image.detach().cpu().squeeze(0), affine=source_t0['image'].affine),
                label=tio.LabelMap(tensor=warped_source_label.squeeze(0).int().detach().cpu(), affine=source_t0['label'].affine),
                flow=tio.ScalarImage(tensor=forward_flow.squeeze(0).detach().cpu(), affine=source_t0['image'].affine)
            )
            warped_subject = reverse_transform(warped_subject)
            dice_metric = DiceMetric(include_background=True)
            dice = dice_metric(tio.OneHot()(warped_subject["label"])[tio.DATA].unsqueeze(0).cpu(),
                               tio.OneHot()(target)[tio.DATA].unsqueeze(0).cpu())

            print(dice)
            dice_metric.reset()
            warped_subject['flow'][tio.DATA] = warped_subject['flow'][tio.DATA] * torch.tensor(source_subject.spacing).view(3, 1, 1, 1)
            warped_subject['image'].save(os.path.join(result_save_path_images, f"warped-t{j}.nii.gz"))
            warped_subject['flow'].save(os.path.join(result_save_path_flows, f"df-t{j}.nii.gz"))
            warped_subject['label'].save(os.path.join(result_save_path_seg, f"warped-t{j}_label.nii.gz"))

# %% Main program
if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description='Beo registration.yaml 3D Longitudinal Images with MLP model')
    parser.add_argument('--dataset_yaml', type=str, help='Path to the dataset yaml file', default="")
    parser.add_argument('--load', type=str, help='Path to the model')
    parser.add_argument('--time_mode', type=str, help='SVF Temporal mode', choices={'svf_mlp', 'svf_lin', 'svf_int'}, default='svf_int')
    args = parser.parse_args()
    test(args=args)

