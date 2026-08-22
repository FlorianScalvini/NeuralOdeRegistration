"""
PyTorch Dataset classes for spatio-temporal longitudinal brain MRI sequences.

Each dataset wraps a list of per-subject session lists, where every session
entry is a ``(image_path, seg_path, age)`` triplet.  At index time the
datasets load all sessions for a given subject, optionally apply spatial
transforms, sort by acquisition age, and return stacked tensors ready for
the registration model.

Two variants are provided:

* :class:`SpatioTemporalDataset` — training dataset.  Filters out subjects
  with fewer than two sessions (single time-points cannot be used for
  longitudinal registration), loads SDF maps alongside images and
  segmentations, and sorts sessions by age before stacking.
* :class:`SpatioTemporalDatasetValidation` — validation / test dataset.
  Simpler pipeline without SDF loading; exposes :meth:`get_subject` so the
  test loop can retrieve the original TorchIO subject (with affine) for
  NIfTI export.

Author : Florian Scalvini
"""

# --- Third-party ---
import numpy as np
import torch
import torchio as tio
import vtk
from torchio import transforms
from vtk.util.numpy_support import vtk_to_numpy


# ──────────────────────────────────────────────────────────────────────────────
#  Training dataset
# ──────────────────────────────────────────────────────────────────────────────

class SpatioTemporalDataset(torch.utils.data.Dataset):
    """Longitudinal MRI dataset for training.

    Loads image volumes, segmentation label maps, signed-distance-function
    (SDF) maps, and acquisition ages for each subject.  Subjects with fewer
    than two sessions are silently discarded because at least two time-points
    are required for longitudinal registration.  Sessions are sorted by age
    before being stacked into tensors.

    Parameters
    ----------
    data : list
        Outer list — one entry per subject.  Inner list — one entry per
        session, each being a ``[image_path, seg_path, age]`` triplet.
        *seg_path* may be ``None`` if no segmentation is available.
    transform : transforms.Transform or None
        Spatial transform applied to each image volume independently.
    transform_seg : transforms.Transform or None
        Spatial transform applied to segmentation and SDF maps independently.
    """

    def __init__(
        self,
        data: list,
        transform: transforms.Transform | None = None,
        augmentation: transforms.Transform | None = None,
        load_surface: bool = False,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.augmentation = augmentation
        self.load_surface = load_surface
        self.data: list = []
        for i in range(len(data)):
            if len(data[i]) >= 2:
                self.data.append(data[i])

    def __len__(self) -> int:
        """Return the number of subjects in the dataset."""
        return len(self.data)


    def __getitem__(
        self, idx: int
    ) -> tuple:
        """Return all sessions for subject *idx* sorted by age.

        Parameters
        ----------
        idx : int
            Subject index.

        Returns
        -------
        mri_stack_out : torch.Tensor
            Stacked MRI volumes of shape ``(T, 1, X, Y, Z)``.
        seg_stack_out : torch.Tensor
            Stacked segmentation label maps of shape ``(T, 1, X, Y, Z)``.
        time_stack_out : torch.Tensor
            Acquisition ages of shape ``(T,)``.
        """
        mri_stack = []
        time_stack = []
        seg_stack = []
        surface_vertices = []
        surface_affines = []
        data = self.data[idx]
        for i in range(len(data)):
            session = tio.Subject(
                image=tio.ScalarImage(data[i][0]),
                label=tio.LabelMap(data[i][1])
            )
            if self.transform is not None:
                session = self.transform(session)
            if self.augmentation is not None:
                session = self.augmentation(session) # type: ignore
            mri_stack.append(session.image.data)
            seg_stack.append(session.label.data)
            time_stack.append(data[i][2])
            if self.load_surface:
                surface_vertices.append(_read_surface_vertices(data[i][3]))
                surface_affines.append(torch.as_tensor(session.image.affine, dtype=torch.float32))
            del session

        # ── 5. stack ──────────────────────────────────────────────────
        mri_stack_out = torch.stack(mri_stack, dim=0)  # (T_total, 1, X, Y, Z)
        seg_stack_out = torch.stack(seg_stack, dim=0)  # (T_total, 1, X, Y, Z)
        time_stack_out = torch.tensor(time_stack, dtype=torch.float)  # (T_total,)

        if self.load_surface:
            return (
                mri_stack_out, seg_stack_out, time_stack_out,
                surface_vertices, surface_affines,
            )
        return mri_stack_out, seg_stack_out, time_stack_out

# ──────────────────────────────────────────────────────────────────────────────
#  Validation / test dataset
# ──────────────────────────────────────────────────────────────────────────────

class SpatioTemporalDatasetValidation(torch.utils.data.Dataset):
    """Longitudinal MRI dataset for validation and testing.

    Lighter variant of :class:`SpatioTemporalDataset` that omits SDF loading
    and keeps all subjects regardless of session count.  Exposes
    :meth:`get_subject` so the test loop can retrieve the full TorchIO
    subject (with affine matrix) for NIfTI-format saving.

    Parameters
    ----------
    data : list
        Outer list — one entry per subject.  Inner list — one entry per
        session, each being a ``[image_path, seg_path, age]`` triplet.
    transform : transforms.Transform or None
        Spatial transform applied to each full subject at load time.
    transform_seg : transforms.Transform or None
        Spatial transform applied to segmentation maps (reserved for API
        consistency; not applied inside ``__getitem__``).
    reverse_transform : transforms.Transform or None
        Inverse spatial transform used to map predictions back to the
        original subject space (e.g. :class:`tio.CropOrPad`).
    """

    def __init__(
        self,
        data: list,
        transform: transforms.Transform | None = None,
        transform_seg: transforms.Transform | None = None,
        reverse_transform: transforms.Transform | None = None,
        load_surface: bool = False,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.transform_seg = transform_seg
        self.data = data
        self.load_surface = load_surface
        resize_size = list(self.transform.transforms[0].target_shape)
        shape_img = tio.ScalarImage(self.data[0][0][0]).shape[1:]
        self.reverse_transform = tio.transforms.Compose([
            tio.transforms.Resize(resize_size),
            tio.transforms.CropOrPad(shape_img)
        ])
    def __len__(self) -> int:
        """Return the number of subjects in the dataset."""
        return len(self.data)

    def get_reverse_transform(self) -> transforms.Transform | None:
        """Return the inverse spatial transform, or ``None`` if not set."""
        return self.reverse_transform

    def get_surface_path(self, subject_idx: int, time_idx: int) -> str:
        """Return the VTK surface path for a validation session."""
        if not self.load_surface:
            raise RuntimeError("surface loading is disabled")
        return self.data[subject_idx][time_idx][3]


    def get_subject(self, idx_subject: int, idx_session: int) -> tio.Subject:
        """Return the raw TorchIO subject at *idx* without applying transforms.

        Parameters
        ----------
        idx_subject : int
            Subject index.
        idx_session : int
            Session index.

        Returns
        -------
        tio.Subject
            Subject loaded from disk with ``image`` and (optionally) ``label``
            fields, preserving the original affine for NIfTI export.
        """
        data = self.data[idx_subject]
        session = tio.Subject(
            image=tio.ScalarImage(data[idx_session][0]),
            label=tio.LabelMap(data[idx_session][1]) if data[idx_session][1] is not None else None,
        )
        return session

    def __getitem__(
        self, idx: int
    ) -> tuple:
        """Return all sessions for subject *idx* as stacked tensors.

        Parameters
        ----------
        idx : int
            Subject index.

        Returns
        -------
        mri_stack_out : torch.Tensor
            Stacked MRI volumes of shape ``(T, 1, X, Y, Z)``.
        seg_stack_out : torch.Tensor
            Stacked segmentation label maps of shape ``(T, 1, X, Y, Z)``,
            or an empty tensor if no labels are available.
        time_stack_out : torch.Tensor
            Acquisition ages of shape ``(T,)``.
        """
        mri_stack = []
        seg_stack = []
        time_stack = []
        surface_vertices = []
        surface_affines = []
        data = self.data[idx]
        for i in range(len(data)):
            session = tio.Subject(
                image=tio.ScalarImage(data[i][0]),
                label=tio.LabelMap(data[i][1]) if data[i][1] is not None else None,

            )
            if self.transform is not None:
                session = self.transform(session)

            mri_stack.append(session.image.data)
            if session.label is not None:
                seg_stack.append(session.label.data)
            time_stack.append(data[i][2])
            if self.load_surface:
                surface_vertices.append(_read_surface_vertices(data[i][3]))
                surface_affines.append(torch.as_tensor(session.image.affine, dtype=torch.float32))
            del session
        mri_stack_out = torch.stack(mri_stack, dim=0)  # (T_total, 1, X, Y, Z)

        if len(seg_stack) > 0:
            seg_stack_out = torch.stack(seg_stack, dim=0)  # (T_total, 1, X, Y, Z)
        else:
            seg_stack_out = torch.empty(0)

        time_stack_out = torch.tensor(time_stack, dtype=torch.float)  # (T_total,)
        if self.load_surface:
            return (
                mri_stack_out, seg_stack_out, time_stack_out,
                surface_vertices, surface_affines,
            )
        return mri_stack_out, seg_stack_out, time_stack_out


def _read_surface_vertices(path: str) -> torch.Tensor:
    """Read mesh points as a native-endian contiguous ``float32`` tensor."""
    if path.lower().endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()
    elif path.lower().endswith(".vtk"):
        reader = vtk.vtkGenericDataObjectReader()
    else:
        raise ValueError(f"surface must end with .vtk or .vtp: {path}")
    reader.SetFileName(path)
    reader.Update()
    mesh = reader.GetOutput()
    if not isinstance(mesh, vtk.vtkPointSet) or mesh.GetPoints() is None:
        raise ValueError(f"surface does not contain a VTK point set: {path}")
    points = np.array(
        vtk_to_numpy(mesh.GetPoints().GetData()),
        dtype=np.float32,
        copy=True,
        order="C",
    )
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"surface vertices must have shape (V,3), got {points.shape}: {path}")
    return torch.from_numpy(points)
