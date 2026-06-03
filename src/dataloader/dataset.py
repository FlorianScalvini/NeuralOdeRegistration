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

class SpatioTemporalDataset(torch.utils.data.dataset.Dataset): # type: ignore
    def __init__(self, data, transform=None, transform_seg=None):
        '''
        PairwiseSubjectsDataset
        :param subjects: Sequence of subjects
        :param transform: Transform composition applying to the subjects
        '''
        super().__init__()
        self.transform = transform
        self.transform_seg = transform_seg
        self.data = []
        for i in range(len(data)):
            if len(data[i]) >= 2:
                self.data.append(data[i])

    def __len__(self):
        '''
            Return the number of subjects in the dataset
        '''
        return len(self.data)


    def __getitem__(self, idx: int):
        '''
            Get the sequence at index idx
            :param idx: index of the sequence
        '''
        mri_stack = []
        seg_stack = []
        time_stack = []
        sdf_stack = []  
        data = self.data[idx]
        for i in range(len(data)):
            session = tio.Subject(
                image=tio.ScalarImage(data[i][0]),
                label=tio.LabelMap(data[i][1]) if data[i][1] is not None else None,
                sdf=tio.ScalarImage(data[i][1].replace("tissue", "sdf_cortex"))
            )
            if self.transform is not None:
                session['image'] = self.transform(session['image'])
            if self.transform_seg is not None:
                session['label'] = self.transform_seg(session['label'])
                session['sdf'] = self.transform_seg(session['sdf'])

            mri_stack.append(session.image.data)
            sdf_stack.append(session.sdf.data)
            if session.label is not None:
                seg_stack.append(session.label.data)
            time_stack.append(data[i][2])
            del session

        # ── 4. sort by age — multi-session first on tie ────────────────
        # stable sort preserves relative order within same age,
        # and multi-session entries come first in all_times (built first)
        # so a stable sort on age keeps multi before mono at equal ages
        sorted_indices = sorted(
            range(len(time_stack)),
            key=lambda i: (time_stack[i]))  # (age) → multi first
        all_sdf = [sdf_stack[i] for i in sorted_indices]
        all_mri = [mri_stack[i] for i in sorted_indices]
        all_seg = [seg_stack[i] for i in sorted_indices]
        all_times = [time_stack[i] for i in sorted_indices]

        # ── 5. stack ──────────────────────────────────────────────────
        mri_stack_out = torch.stack(all_mri, dim=0)  # (T_total, 1, X, Y, Z)
        sdf_stack_out = torch.stack(all_sdf, dim=0)  # (T_total, 1, X, Y, Z)
        seg_stack_out = torch.stack(all_seg, dim=0)  # (T_total, 1, X, Y, Z)
        time_stack_out = torch.tensor(all_times, dtype=torch.float)  # (T_total,)
        
        return mri_stack_out, seg_stack_out, time_stack_out, sdf_stack_out




class SpatioTemporalDatasetValidation(torch.utils.data.dataset.Dataset): # type: ignore
    def __init__(self, data, transform=None, transform_seg=None, reverse_transform=None):
        '''
        PairwiseSubjectsDataset
        :param subjects: Sequence of subjects
        :param transform: Transform composition applying to the subjects
        '''
        super().__init__()
        self.transform = transform
        self.transform_seg = transform_seg
        self.reverse_transform = reverse_transform
        self.data = data

    def __len__(self):
        '''
            Return the number of subjects in the dataset
        '''
        
        return len(self.data)

    def get_reverse_transform(self) -> transforms.Transform | None:
        return self.reverse_transform
    
    def get_subject(self, idx: int) -> tio.Subject:
        data = self.data[idx]
        session = tio.Subject(
            image=tio.ScalarImage(data[0][0]),
            label=tio.LabelMap(data[0][1]) if data[0][1] is not None else None
        )
        return session

    def __getitem__(self, idx: int):
        '''
            Get the sequence at index idx
            :param idx: index of the sequence
        '''
        mri_stack = []
        seg_stack = []
        time_stack = []
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
            del session
        mri_stack_out = torch.stack(mri_stack, dim=0)  # (T_total, 1, X, Y, Z)

        if len(seg_stack) > 0:
            seg_stack_out = torch.stack(seg_stack, dim=0)  # (T_total, 1, X, Y, Z)
        else:
            seg_stack_out = torch.empty(0)

        time_stack_out = torch.tensor(time_stack, dtype=torch.float)  # (T_total,)
        return mri_stack_out, seg_stack_out, time_stack_out
