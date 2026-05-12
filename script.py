
import json
import os
import torch
import monai
import torchio as tio
#ANTS 0.83
#Nothing 0.72
#Our 0.75
dice_metric = monai.metrics.DiceMetric()
with open("/home/florian/Documents/Dataset/Calgary/data_val.json", 'r') as f:
    # Parsing the JSON file into a Python dictionary
    data = json.load(f)
output_dir = "./temp/"
os.makedirs(output_dir, exist_ok=True)
for subject in data['subjects']:
    target_seg = "/home/florian/Documents/Dataset/Calgary/" + subject['sessions'][-1]['segmentation']
    pred_seg = "/home/florian/Documents/Dataset/Calgary/" + subject['sessions'][0]['segmentation']
    print(target_seg, pred_seg)
    if target_seg == pred_seg:
        continue
    pred_seg = "/home/florian/Documents/Programs/longitudinal-svf/src/others/ants/temp/" + subject['subject_id'] + "_SyN_gs0.7_seg__Warped_label.nii.gz"
    pred = tio.LabelMap(pred_seg)
    target = tio.LabelMap(target_seg)
    one_hot = tio.OneHot()
    pred_one_hot = one_hot(pred)
    target_one_hot = one_hot(target)
    dice_metric(pred_one_hot.data.unsqueeze(0), target_one_hot.data.unsqueeze(0))
    print(subject['subject_id'], torch.mean(dice_metric.get_buffer()[-1]).item())
print(dice_metric.aggregate(reduction='mean').item())



