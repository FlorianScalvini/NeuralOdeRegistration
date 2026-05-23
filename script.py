
import json
import os
import torch
import monai
import torchio as tio
#ANTS 0.83
#Nothing 0.72
#Our 0.75

looktable = {
    0: "BG",
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    7: 5,
    8: 6,
    10: 7,
    11: 8,
    12: 9,
    13: 10,
    14: 11,
    15: 12,
    16: 13,
    17: 14,
    18: 15,
    24: 16,
    26: 17,
    41: 1,
    42: 2,
    43: 3,
    44: 4,
    46: 5,
    47: 6,
    49: 7,
    50: 8,
    51: 9,
    52: 10,
    53: 14,
    54: 15,
    58: 17,
    77: 18,
    85: 19,
    819: 20,
    820: 20,
    821: 20,
    822: 20,
    843: 20,
    844: 20,
    865: 20,
    866: 20,
    869: 20,
    870: 20,
}

looktable = {
    0: "BG",
    1: "WM",
    2: "cortex",
    3: "Ventricules lateraux",
    4: "Ventricules lateraux",
    5: "Cerebellum WM",
    6: "Cerebellum",
    7: "",
    8: "Caudate",
    10: "thalamus",
    11: "3rd ventricule",
    12: "4th ventricule",
    13: "Pallidum",
    14: "Hippocampus",
    15: "Amygdala",
    16: "Brain stem",
    17: 14,
    18: 15,
    19:
    20
}


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
    #pred_seg = "/home/florian/Documents/Programs/longitudinal-svf/src/others/ants/temp/" + subject['subject_id'] + "_SyN_gs0.7_seg__Warped_label.nii.gz"
    pred = tio.LabelMap(pred_seg)
    target = tio.LabelMap(target_seg)
    one_hot = tio.OneHot()
    pred_one_hot = one_hot(pred)
    target_one_hot = one_hot(target)
    dice_metric(pred_one_hot.data.unsqueeze(0), target_one_hot.data.unsqueeze(0))
    print(dice_metric.get_buffer()[-1])
    print(subject['subject_id'], torch.mean(dice_metric.get_buffer()[-1]).item())
print(dice_metric.aggregate(reduction='mean').item())



