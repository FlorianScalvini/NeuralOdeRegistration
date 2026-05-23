import os.path

import torch
import argparse
import torchio as tio
from registration_svf.registration import RegistrationModule
from registration_svf.modules.unet import DyNUnet
from registration_svf.utils.grid_utils import warp
import yaml
import pandas as pd

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

def inference(source : tio.Subject, target : tio.Subject, model : RegistrationModule, device: str):
    model.eval().to(device)

    source_img = source['image'][tio.DATA].unsqueeze(0).to(device)
    target_img = target['image'][tio.DATA].unsqueeze(0).to(device)
    with torch.no_grad():
        velocity = model(torch.cat([source_img, target_img], dim=1))
    forward_flow = model.velocity2displacement(velocity)
    backward_flow = model.velocity2displacement(-velocity)
    return forward_flow, backward_flow

def main(img_src, img_target, lbl_src, lbl_target, csize, rsize, num_classes, load, savePath = './', index=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result_save_path_images = os.path.join(savePath, "images")
    result_save_path_flows = os.path.join(savePath, "flows")
    result_save_path_seg = os.path.join(savePath, "parcellations")
    os.makedirs(result_save_path_images, exist_ok=True)
    os.makedirs(result_save_path_seg, exist_ok=True)
    os.makedirs(result_save_path_flows, exist_ok=True)

    ## Config Subject
    source_subject = tio.Subject(
        image=tio.ScalarImage(img_src),
        label=tio.LabelMap(lbl_src),
    )

    target_subject = tio.Subject(
        image=tio.ScalarImage(img_target),
        label=tio.LabelMap(lbl_target),
    )

    transforms_input = tio.Compose([
        tio.CropOrPad(target_shape=csize),
        tio.Resize(target_shape=rsize),
        tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5), masking_method='label'),
        #tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5), include=['image']),
        tio.transforms.Clamp(out_min=0, out_max=1, include=['image'])
    ])

    transforms_without_norm = tio.Compose([
        tio.CropOrPad(target_shape=csize),
        tio.Resize(target_shape=rsize),
        tio.OneHot()
    ])

    reverse_transform = tio.Compose([
        tio.Resize(target_shape=csize),
        tio.CropOrPad(target_shape=source_subject["image"][tio.DATA].shape[1:]),
    ])

    source_subject_transformed = transforms_input(source_subject)
    target_subject_transformed = transforms_input(target_subject)
    source_subject_warp_transformed = transforms_without_norm(source_subject)
    target_subject_warp_transformed = transforms_without_norm(target_subject)

    model = DyNUnet(
        in_channels=2,
        out_channels=3,
        kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        strides=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]).eval().to(device)

    model = RegistrationModule(
        model=model,
        int_steps=9).eval().to(device)
    model.load_state_dict(torch.load(load))

    forward_flow, backward_flow = inference(source_subject_transformed, target_subject_transformed, model, device)

    source_img = source_subject_warp_transformed['image'][tio.DATA].unsqueeze(0).to(device).float()
    source_label = source_subject_warp_transformed['label'][tio.DATA].unsqueeze(0).to(device).float()
    warped_source_img = warp(source_img, forward_flow)
    warped_source_label = warp(source_label, forward_flow)

    warped_subjects = tio.Subject(
        warped_source_img = tio.ScalarImage(tensor=warped_source_img.squeeze(0).cpu().detach(), affine=source_subject_transformed['image'].affine),
        warped_source_label = tio.ScalarImage(tensor=torch.argmax(warped_source_label, dim=1).cpu().detach(), affine=source_subject_transformed['label'].affine),
        flow=tio.ScalarImage(tensor=forward_flow.squeeze(0).detach().cpu(), affine=source_subject_transformed['image'].affine)
    )

    print(os.path.join(result_save_path_flows, f"df-t{index}.nii.gz"))
    warped_subjects = reverse_transform(warped_subjects)
    warped_subjects['flow'][tio.DATA] = warped_subjects['flow'][tio.DATA] * torch.tensor(source_subject.spacing).view(3, 1, 1, 1)
    warped_subjects['warped_source_img'].save(os.path.join(result_save_path_images, f"warped-t{index}.nii.gz"))
    warped_subjects['flow'].save(os.path.join(result_save_path_flows, f"df-t{index}.nii.gz"))
    warped_subjects['warped_source_label'].save(os.path.join(result_save_path_seg, f"warped-t{index}_label.nii.gz"))



if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description='Inference Registration 3D Images')
    parser.add_argument('--dataset_yaml', type=str, help='Path to the source image', required=True)
    parser.add_argument('--savePath', type=str, help='Path to the output directory', required=True)
    parser.add_argument("--load", type=str, help='Path to the model weights', required=True)
    args = parser.parse_args()

    with open(args.dataset_yaml, "r") as f:
        config = yaml.safe_load(f)
    rsize = config['rsize']
    csize = config['csize']
    num_classes = config['num_classes']
    df = pd.read_csv(config['csv_path'])
    t0 = int(config['t0'])
    lst_images = []
    lst_labels = []
    for index, row in df.iterrows():
        lst_images.append(row['image'])
        lst_labels.append(row['label'])
    image_source = lst_images[t0]
    label_source = lst_labels[t0]
    lst_labels = lst_labels[:t0] + lst_labels[t0+1:]
    lst_images = lst_images[:t0] + lst_images[t0+1:]
    output_path = os.path.join(args.savePath, args.name)
    os.makedirs(output_path, exist_ok=True)
    for i in range(len(lst_images)):
        print(f"Processing {i+1} / {len(lst_images)} : Image: {lst_images[i]}, Label: {lst_labels[i]}")
        main(image_source, lst_images[i], label_source, lst_labels[i], csize, rsize, num_classes, args.load, savePath=output_path, index=i)
