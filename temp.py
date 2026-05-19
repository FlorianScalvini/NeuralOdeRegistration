import torch
import torchio as tio
import random
import scipy.ndimage as ndi
import numpy as np
import glob

list_seg = glob.glob("/home/florian/Documents/Dataset/dHCP/Atlas/parcellations/*.nii.gz")
for path in list_seg:
    seg = tio.Subject(
        seg=tio.LabelMap(path)
    )
    filename = path.split("/")[-1]
    seg_data = seg.seg.data.squeeze().numpy()  # Remove channel dimension and convert to numpy

    # Get voxel spacing (in mm)
    spacing = seg.seg.spacing  # (x, y, z) spacing in mm
    print(f"Voxel spacing: {spacing} mm")

    # Create binary mask for class 2
    class_2_mask = (seg_data == 4).astype(np.uint8) + (seg_data == 3).astype(np.uint8)

    # Calculate signed distance transform in voxel units
    distance_outside = ndi.distance_transform_edt(1 - class_2_mask)
    distance_inside = ndi.distance_transform_edt(class_2_mask)

    # Combine to create signed distance function (in voxel units)
    sdf_voxels = distance_outside - distance_inside

    # Convert to physical units (mm) using voxel spacing
    # Use isotropic approximation or directional spacing
    spacing_array = np.array(spacing)
    avg_spacing = np.mean(spacing_array)  # Average spacing for isotropic approximation
    sdf_mm = sdf_voxels * avg_spacing

    # Clamp to [-3mm, +3mm]
    sdf_clamped = np.clip(sdf_mm, -3.0, 3.0)

    print(f"SDF shape: {sdf_clamped.shape}")
    print(f"SDF range before clamp: [{sdf_mm.min():.2f}, {sdf_mm.max():.2f}] mm")
    print(f"SDF range after clamp: [{sdf_clamped.min():.2f}, {sdf_clamped.max():.2f}] mm")
    print(f"Number of class 2 voxels: {class_2_mask.sum()}")

    # Convert back to torch tensor
    sdf_tensor = torch.from_numpy(sdf_clamped / 3).float()

    # Optionally save as a new TorchIO image with the same affine/spacing
    sdf_image = tio.ScalarImage(tensor=sdf_tensor.unsqueeze(0), affine=seg.seg.affine)
    sdf_image.save(path.replace("tissue", "sdf_cortex"))