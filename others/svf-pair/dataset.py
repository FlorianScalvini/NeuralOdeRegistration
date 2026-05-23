import torch
import torchio as tio
from typing import Sequence
import random
import itertools

import os
import glob
import torch
import torchio as tio
from typing import Sequence
import csv
import pandas as pd
import json
from torchio import transforms

class SpatioTemporalDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data, transform=None, transform_seg=None):
        '''
        PairwiseSubjectsDataset
        :param subjects: Sequence of subjects
        :param transform: Transform composition applying to the subjects
        '''
        super().__init__()
        self.transform = transform
        self.transform_seg = transform_seg
        self.data = data

            # IGNORE >3 (val)


    def __len__(self):
        '''
            Return the number of subjects in the dataset
        '''
        return len(self.data)

    def get_subject_affine(self, idx: int) -> torch.Tensor:
        data = self.data[idx]
        session = tio.Subject(
            image=tio.ScalarImage(data[0][0]),
            label=tio.LabelMap(data[0][1]) if data[0][1] is not None else None
        )
        return session['image'].data.affine

    def __getitem__(self, idx: int) -> tio.Subject:
        '''
            Get the sequence at index idx
            :param idx: index of the sequence
        '''

        mri_stack = []
        seg_stack = []
        data = self.data[idx]
        for i in range(len(data)):
            session = tio.Subject(
                image=tio.ScalarImage(data[i][0]),
                label=tio.LabelMap(data[i][1]) if data[i][1] is not None else None,

            )
            if self.transform is not None:
                session = self.transform(session)
            if self.transform_seg is not None:
                session = self.transform_seg(session)

            mri_stack.append(session.image.data)
            if session.label is not None:
                seg_stack.append(session.label.data)
            del session

        # ── 5. stack ──────────────────────────────────────────────────
        #mri_stack.reverse()
        #seg_stack.reverse()
        #time_stack.reverse()
        mri_stack_out = torch.stack(mri_stack, dim=0)  # (T_total, 1, X, Y, Z)

        if len(seg_stack) > 0:
            seg_stack_out = torch.stack(seg_stack, dim=0)  # (T_total, 1, X, Y, Z)
        else:
            seg_stack_out = torch.empty(0)

        # is_mono_out[i] = False → real multi-session timepoint
        # is_mono_out[i] = True  → mono-session subject (no NCC supervision)

        return mri_stack_out, seg_stack_out



class SpatioTemporalDatasetValidation(torch.utils.data.dataset.Dataset):
    def __init__(self, data, transform=None, transform_seg=None):
        '''
        PairwiseSubjectsDataset
        :param subjects: Sequence of subjects
        :param transform: Transform composition applying to the subjects
        '''
        super().__init__()
        self.transform = transform
        self.transform_seg = transform_seg
        self.data = data

            # IGNORE >3 (val)


    def __len__(self):
        '''
            Return the number of subjects in the dataset
        '''
        return len(self.data)

    def get_subject_affine(self, idx: int) -> torch.Tensor:
        data = self.data[idx]
        session = tio.Subject(
            image=tio.ScalarImage(data[0][0]),
            label=tio.LabelMap(data[0][1]) if data[0][1] is not None else None
        )
        return session['image'].data.affine

    def __getitem__(self, idx: int) -> tio.Subject:
        '''
            Get the sequence at index idx
            :param idx: index of the sequence
        '''

        mri_stack = []
        seg_stack = []
        data = self.data[idx]
        for i in range(len(data)):
            session = tio.Subject(
                image=tio.ScalarImage(data[i][0]),
                label=tio.LabelMap(data[i][1]) if data[i][1] is not None else None,

            )
            if self.transform is not None:
                session = self.transform(session)
            if self.transform_seg is not None:
                session = self.transform_seg(session)

            mri_stack.append(session.image.data)
            if session.label is not None:
                seg_stack.append(session.label.data)
            del session

        # ── 5. stack ──────────────────────────────────────────────────
        #mri_stack.reverse()
        #seg_stack.reverse()
        #time_stack.reverse()
        mri_stack_out = torch.stack(mri_stack, dim=0)  # (T_total, 1, X, Y, Z)

        if len(seg_stack) > 0:
            seg_stack_out = torch.stack(seg_stack, dim=0)  # (T_total, 1, X, Y, Z)
        else:
            seg_stack_out = torch.empty(0)

        # is_mono_out[i] = False → real multi-session timepoint
        # is_mono_out[i] = True  → mono-session subject (no NCC supervision)

        return mri_stack_out, seg_stack_out
