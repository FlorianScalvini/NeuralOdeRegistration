import json
import numpy as np
import os
import torchio as tio
# pip install vtk
import os

import math
import vtk
import glob
import argparse
import meshio

data_val_path = "/home/florian/Documents/Dataset/Calgary/data_val.json"
data_result_path = "/home/florian/PyCharmMiscProject/src/calgary_longitudinal_subject/26_29_13_32/result.json"
rootpath = "/home/florian/Documents/Dataset/Calgary/"
with open(data_val_path, 'r') as f:
    # Parsing the JSON file into a Python dictionary
    data_vals = json.load(f)

with open(data_result_path, 'r') as f:
    # Parsing the JSON file into a Python dictionary
    data_results = json.load(f)




def _clone_identity(img):
    out = vtk.vtkImageData()
    out.ShallowCopy(img)
    out.SetOrigin(0, 0, 0)
    out.SetSpacing(1, 1, 1)
    return out


def _wsinc_params(s):
    # passband = 10^(-4*s), iterations = 20 + 40*s
    return 10.0 ** (-4.0 * s), int(round(20 + 40 * s))


def _surf_nets_iters(s):
    return int(math.floor(15.0 * s * s + 9.0 * s))


def _gen_surface_ijk(img, labels, method="flying_edges", smoothing_factor=0.5, surf_nets_internal=False,
                     decimation=0.0):
    ijk = _clone_identity(img)

    if method == "flying_edges":
        fe = vtk.vtkDiscreteFlyingEdges3D()
        fe.SetInputData(ijk)
        fe.ComputeGradientsOff()
        fe.ComputeNormalsOff()
        for i, v in enumerate(labels):
            fe.SetValue(i, int(v))
        fe.Update()
        surf = fe.GetOutput()
    elif method == "surface_nets":
        sn = vtk.vtkSurfaceNets3D()
        sn.SetInputData(ijk)
        sn.SmoothingOff()
        if surf_nets_internal:
            sn.SmoothingOn()
            sn.SetNumberOfIterations(_surf_nets_iters(smoothing_factor))
        for i, v in enumerate(labels):
            sn.SetValue(i, int(v))
        sn.Update()
        surf = sn.GetOutput()
    else:
        raise ValueError("method must be 'flying_edges' or 'surface_nets'")

    if surf.GetNumberOfPolys() == 0:
        empty = vtk.vtkPolyData();
        empty.Initialize();
        return empty

    if decimation and decimation > 0.0:
        dec = vtk.vtkDecimatePro()
        dec.SetInputData(surf)
        dec.SetFeatureAngle(60);
        dec.SplittingOff();
        dec.PreserveTopologyOn();
        dec.SetMaximumError(1)
        dec.SetTargetReduction(float(decimation))
        dec.Update()
        surf = dec.GetOutput()

    if smoothing_factor > 0.0 and not surf_nets_internal:
        pb, iters = _wsinc_params(smoothing_factor)
        sm = vtk.vtkWindowedSincPolyDataFilter()
        sm.SetInputData(surf)
        sm.SetNumberOfIterations(iters)
        sm.SetPassBand(pb)
        sm.BoundarySmoothingOff();
        sm.FeatureEdgeSmoothingOff()
        sm.NonManifoldSmoothingOn();
        sm.NormalizeCoordinatesOn()
        sm.Update()
        surf = sm.GetOutput()

    return surf


def _ijk_to_world(surface_ijk, img_from_reader):
    # Try direction matrix (VTK 9), else build from origin/spacing
    m = vtk.vtkMatrix4x4();
    m.Identity()
    if hasattr(img_from_reader, "GetDirectionMatrix"):
        dm = img_from_reader.GetDirectionMatrix()
        sx, sy, sz = img_from_reader.GetSpacing()
        for r in range(3):
            for c in range(3):
                m.SetElement(r, c, dm.GetElement(r, c) * (sx if c == 0 else sy if c == 1 else sz))
        ox, oy, oz = img_from_reader.GetOrigin()
        m.SetElement(0, 3, ox);
        m.SetElement(1, 3, oy);
        m.SetElement(2, 3, oz)
    else:
        sx, sy, sz = img_from_reader.GetSpacing()
        ox, oy, oz = img_from_reader.GetOrigin()
        m.SetElement(0, 0, sx);
        m.SetElement(1, 1, sy);
        m.SetElement(2, 2, sz)
        m.SetElement(0, 3, ox);
        m.SetElement(1, 3, oy);
        m.SetElement(2, 3, oz)

    xf = vtk.vtkTransform();
    xf.SetMatrix(m)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(surface_ijk);
    tf.SetTransform(xf);
    tf.Update()
    return tf.GetOutput()


def _finish(surface_world, compute_normals, method):
    out = vtk.vtkPolyData()
    if compute_normals and method == "flying_edges":
        n = vtk.vtkPolyDataNormals()
        n.SetInputData(surface_world)
        n.ConsistencyOn();
        n.SplittingOff()
        n.Update()
        out.ShallowCopy(n.GetOutput())
    else:
        out.ShallowCopy(surface_world)
    pd = out.GetPointData()
    if pd is not None:
        pd.RemoveArray("ImageScalars")
    return out


# ---------- new: fuse multiple labels to one binary mask ----------
def _fuse_labels_to_binary(img, labels):
    labels = [int(v) for v in labels]
    labels = sorted(set(labels))
    # start with first mask
    th = vtk.vtkImageThreshold()
    th.SetInputData(img)
    th.ThresholdBetween(labels[0], labels[0])
    th.SetInValue(1);
    th.SetOutValue(0)
    th.SetOutputScalarTypeToUnsignedChar()
    th.Update()
    merged = th.GetOutput()
    # OR the rest
    for val in labels[1:]:
        t2 = vtk.vtkImageThreshold()
        t2.SetInputData(img)
        t2.ThresholdBetween(val, val)
        t2.SetInValue(1);
        t2.SetOutValue(0)
        t2.SetOutputScalarTypeToUnsignedChar()
        t2.Update()
        logic = vtk.vtkImageLogic()
        logic.SetInput1Data(merged);
        logic.SetInput2Data(t2.GetOutput())
        logic.SetOperationToOr();
        logic.SetOutputTrueValue(1)
        logic.Update()
        merged = logic.GetOutput()
    return merged


def _write_binary_mask_nifti(mask_uc, ref_img, out_path):
    """
    Save the fused binary mask (unsigned char) as NIfTI, copying geometry.
    """
    # Ensure geometry matches reference
    mask_uc.SetOrigin(ref_img.GetOrigin())
    mask_uc.SetSpacing(ref_img.GetSpacing())
    if hasattr(ref_img, "GetDirectionMatrix") and hasattr(mask_uc, "SetDirectionMatrix"):
        mask_uc.SetDirectionMatrix(ref_img.GetDirectionMatrix())

    w = vtk.vtkNIFTIImageWriter()
    w.SetFileName(out_path)
    w.SetInputData(mask_uc)

    # Build sform/qform from direction+spacing+origin (good enough for most pipelines)
    m = vtk.vtkMatrix4x4();
    m.Identity()
    if hasattr(ref_img, "GetDirectionMatrix"):
        dm = ref_img.GetDirectionMatrix()
        sx, sy, sz = ref_img.GetSpacing()
        for r in range(3):
            for c in range(3):
                m.SetElement(r, c, dm.GetElement(r, c) * (sx if c == 0 else sy if c == 1 else sz))
        ox, oy, oz = ref_img.GetOrigin()
        m.SetElement(0, 3, ox);
        m.SetElement(1, 3, oy);
        m.SetElement(2, 3, oz)
    w.SetSFormMatrix(m)
    w.SetQFormMatrix(m)

    w.Write()
    print(f"Saved fused mask NIfTI: {out_path}")


# ---------- existing: single-label → vtk ----------
def convert_nifti_label_to_vtk(
        nii_path, label_value, out_vtk,
        decimation_factor=0.0, smoothing_factor=0.5,
        compute_surface_normals=True,
        method="flying_edges",  # "flying_edges" or "surface_nets"
        surf_nets_internal_smoothing=False
):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nii_path);
    reader.Update()
    img = reader.GetOutput()

    surf_ijk = _gen_surface_ijk(
        img, [int(label_value)],
        method=method,
        smoothing_factor=smoothing_factor,
        surf_nets_internal=surf_nets_internal_smoothing,
        decimation=decimation_factor
    )
    surf_world = _ijk_to_world(surf_ijk, img)
    final = _finish(surf_world, compute_surface_normals, method)

    ug = _polydata_to_unstructured_grid(final)
    cast_all_arrays_to_float32(ug)
    w = vtk.vtkUnstructuredGridWriter()
    w.SetFileName(out_vtk)
    w.SetInputData(ug)
    w.Write()


def _polydata_to_unstructured_grid(poly: vtk.vtkPolyData) -> vtk.vtkUnstructuredGrid:
    # Triangulate (safer for conversion)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()

    # AppendFilter converts any vtkDataSet to vtkUnstructuredGrid
    app = vtk.vtkAppendFilter()
    app.AddInputData(tri.GetOutput())
    app.Update()

    ug = vtk.vtkUnstructuredGrid()
    ug.ShallowCopy(app.GetOutput())
    return ug


# ---------- new: multi-label union → vtk (and optional NIfTI mask) ----------
def convert_nifti_labels_union_to_vtk(
        nii_path, labels, out_vtk,
        out_mask_nifti=None,  # if set, also saves fused mask as NIfTI
        new_label_value=1,
        decimation_factor=0.0, smoothing_factor=0.5,
        compute_surface_normals=True,
        method="flying_edges",
        surf_nets_internal_smoothing=False
):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nii_path);
    reader.Update()
    img = reader.GetOutput()

    merged_mask_uc = _fuse_labels_to_binary(img, labels)

    # optionally write the fused binary mask to disk
    if out_mask_nifti:
        _write_binary_mask_nifti(merged_mask_uc, img, out_mask_nifti)

    # map 1 -> new_label_value for surface extraction
    cast = vtk.vtkImageShiftScale()
    cast.SetInputData(merged_mask_uc)
    cast.SetShift(0.0);
    cast.SetScale(float(new_label_value))
    cast.SetOutputScalarTypeToUnsignedShort()
    cast.Update()
    fused_labelmap = cast.GetOutput()

    # preserve geometry (again, for safety)
    fused_labelmap.SetOrigin(img.GetOrigin())
    fused_labelmap.SetSpacing(img.GetSpacing())
    if hasattr(img, "GetDirectionMatrix") and hasattr(fused_labelmap, "SetDirectionMatrix"):
        fused_labelmap.SetDirectionMatrix(img.GetDirectionMatrix())

    # extract single union surface
    surf_ijk = _gen_surface_ijk(
        fused_labelmap, [new_label_value],
        method=method,
        smoothing_factor=smoothing_factor,
        surf_nets_internal=surf_nets_internal_smoothing,
        decimation=decimation_factor
    )
    surf_world = _ijk_to_world(surf_ijk, fused_labelmap)
    final = _finish(surf_world, compute_surface_normals, method)

    ug = _polydata_to_unstructured_grid(final)
    cast_all_arrays_to_float32(ug)
    w = vtk.vtkUnstructuredGridWriter()
    w.SetFileName(out_vtk)  # keep .vtk
    w.SetInputData(ug)
    w.Write()


def cast_all_arrays_to_float32(ug: vtk.vtkUnstructuredGrid):
    pd = ug.GetPointData()
    cd = ug.GetCellData()
    for data in (pd, cd):
        names = [data.GetArrayName(i) for i in range(data.GetNumberOfArrays())]
        arrays = [data.GetArray(i) for i in range(data.GetNumberOfArrays())]
        # remove all first to avoid index shifts
        for name in names:
            data.RemoveArray(name)
        # re-add cast copies
        for name, arr in zip(names, arrays):
            fa = vtk.vtkFloatArray()
            fa.SetName(name if name is not None else "")
            fa.SetNumberOfComponents(arr.GetNumberOfComponents())
            fa.SetNumberOfTuples(arr.GetNumberOfTuples())
            for i in range(arr.GetNumberOfTuples()):
                fa.SetTuple(i, arr.GetTuple(i))
            data.AddArray(fa)


# ---------- optional: export *all* labels present ----------
def export_all_labels_to_vtk(nii_path, out_prefix, **kwargs):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nii_path);
    reader.Update()
    img = reader.GetOutput()

    acc = vtk.vtkImageAccumulate()
    acc.SetInputData(img)
    acc.IgnoreZeroOn()
    rng = img.GetScalarRange()
    low, high = int(math.floor(rng[0])), int(math.ceil(rng[1]))
    acc.SetComponentOrigin(0, 0, 0)
    acc.SetComponentSpacing(1, 1, 1)
    acc.SetComponentExtent(low, high, 0, 0, 0, 0)
    acc.Update()
    scal = acc.GetOutput().GetPointData().GetScalars()
    labels = [v for i, v in enumerate(range(low, high + 1)) if scal.GetTuple1(i) > 0]

    for lv in labels:
        out = f"{out_prefix}_label{lv}.vtk"
        convert_nifti_label_to_vtk(nii_path, lv, out, **kwargs)

# If input mesh is .stl
#######################
def rescale_initial_smooth_mesh_to_folded_mesh(initial_smooth_mesh, folded_mesh):
    """
    Scaling up the initial smooth mesh, taken as the convex hull, to the size of the folded mesh.
    Args:
    - initial_smooth_mesh: .stl mesh
    - folded_mesh: .stl mesh
    Output:
    - (rescaled) initial_smooth_mesh: .stl mesh
    """

    L1 = (max(folded_mesh.points[:, 0]) - min(folded_mesh.points[:, 0])) / (
                max(initial_smooth_mesh.points[:, 0]) - min(initial_smooth_mesh.points[:, 0]))
    L2 = (max(folded_mesh.points[:, 1]) - min(folded_mesh.points[:, 1])) / (
                max(initial_smooth_mesh.points[:, 1]) - min(initial_smooth_mesh.points[:, 1]))
    L3 = (max(folded_mesh.points[:, 2]) - min(folded_mesh.points[:, 2])) / (
                max(initial_smooth_mesh.points[:, 2]) - min(initial_smooth_mesh.points[:, 2]))
    initial_smooth_mesh.points[:, 0] = L1 * initial_smooth_mesh.points[:, 0]
    initial_smooth_mesh.points[:, 1] = L2 * initial_smooth_mesh.points[:, 1]
    initial_smooth_mesh.points[:, 2] = L3 * initial_smooth_mesh.points[:, 2]

    return initial_smooth_mesh



def compute_gyrification_index(rescaled_initial_smooth_bmesh, folded_bmesh):
    """
    Args:
    - initial_smooth_mesh: .stl boundary mesh
    - folded_mesh: .stl mesh
    Output:
    - GI: gyrification index (int)
    """
    ### Convex hull
    Area_convex_hull = 0.0  # = area of the initial unfolded mesh
    """
    Ntmp = np.cross(initial_smooth_mesh.points[initial_smooth_mesh.cells_dict['triangle'][:,1]] - initial_smooth_mesh.points[initial_smooth_mesh.cells_dict['triangle'][:,0]], 
                    initial_smooth_mesh.points[initial_smooth_mesh.cells_dict['triangle'][:,2]] - initial_smooth_mesh.points[initial_smooth_mesh.cells_dict['triangle'][:,0]])
    """
    for face in rescaled_initial_smooth_bmesh.cells_dict["triangle"]:  # e.g. triangle=[0, 2, 4], with 0, 2, 4 node indices
        Ntmp = np.cross(rescaled_initial_smooth_bmesh.points[face[1]] - rescaled_initial_smooth_bmesh.points[face[0]],
                        rescaled_initial_smooth_bmesh.points[face[2]] - rescaled_initial_smooth_bmesh.points[face[0]])
        Area_convex_hull += 0.5 * np.linalg.norm(Ntmp)

    ### Folded mesh
    Area_folded_mesh = 0.0
    """
    Ntmp_2 = np.cross(rescaled_folded_mesh.points[rescaled_folded_mesh.cells_dict['triangle'][:,1]] - rescaled_folded_mesh.points[rescaled_folded_mesh.cells_dict['triangle'][:,0]], 
                      rescaled_folded_mesh.points[rescaled_folded_mesh.cells_dict['triangle'][:,2]] - rescaled_folded_mesh.points[rescaled_folded_mesh.cells_dict['triangle'][:,0]])
    """
    for face in folded_bmesh.cells_dict["triangle"]:
        Ntmp_2 = np.cross(folded_bmesh.points[face[1]] - folded_bmesh.points[face[0]],
                          folded_bmesh.points[face[2]] - folded_bmesh.points[face[0]])
        Area_folded_mesh += 0.5 * np.linalg.norm(Ntmp_2)

    GI = Area_folded_mesh / Area_convex_hull

    return GI



for idx in range(len(data_vals["subjects"])):
    print('New subject')
    subject = data_vals["subjects"][idx]
    subject_val = data_vals["subjects"][idx]
    os.makedirs("./temp/", exist_ok=True)
    vtk_path_init = "./temp/vtk_temp_init.vtk"
    img_init = rootpath + subject["sessions"][0]['segmentation']
    convert_nifti_labels_union_to_vtk(img_init, [2], vtk_path_init)
    initial_smooth_mesh = meshio.read(vtk_path_init)
    for idx_session in range(1, len(subject["sessions"])):
        os.makedirs("./temp/", exist_ok=True)
        vtk_path = "./temp/vtk_temp.vtk"
        vtk_path_val = "./temp/vtk_temp_val.vtk"
        img = rootpath + subject["sessions"][idx_session]['segmentation']
        img_val = f"/home/florian/PyCharmMiscProject/src/calgary_longitudinal_subject/26_29_13_40/seg_{idx}_{idx_session}.nii.gz"
        convert_nifti_labels_union_to_vtk(img, [2], vtk_path)
        convert_nifti_labels_union_to_vtk(img_val, [2], vtk_path_val)
        folded_mesh = meshio.read(vtk_path)
        # rescale initial smooth brain mesh onto the folded brain mesh
        rescaled_initial_smooth_mesh = rescale_initial_smooth_mesh_to_folded_mesh(initial_smooth_mesh, folded_mesh)
        GI = compute_gyrification_index(rescaled_initial_smooth_mesh, folded_mesh)

        folded_mesh_val = meshio.read(vtk_path_val)
        # rescale initial smooth brain mesh onto the folded brain mesh
        rescaled_initial_smooth_mesh = rescale_initial_smooth_mesh_to_folded_mesh(initial_smooth_mesh, folded_mesh)
        GI_val = compute_gyrification_index(rescaled_initial_smooth_mesh, folded_mesh)
        print(f"Subject {idx}, session {idx_session} -- Target GI : {GI}, Segmentation def : {GI_val}")
