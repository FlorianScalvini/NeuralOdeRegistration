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
    def __init__(self, data, transform=None):
        '''
        PairwiseSubjectsDataset
        :param subjects: Sequence of subjects
        :param transform: Transform composition applying to the subjects
        '''
        super().__init__()
        self.transform = transform
        self.discriminator_phase = False
        self.length_fake_sequence = 5
        self.multi_session = []
        self.single_session = []
        for i in range(len(data)):
            if len(data[i]) == 1:
                self.single_session.append(data[i])
            elif len(data[i]) >= 2:
                self.multi_session.append(data[i])
            #IGNORE >3 (val)

    def __len__(self):
        '''
            Return the number of subjects in the dataset
        '''
        return len(self.multi_session)


    def __getitem__(self, idx: int) -> tio.Subject:
        '''
            Get the sequence at index idx
            :param idx: index of the sequence
        '''
        mri_stack = []
        seg_stack = []
        time_stack = []
        data = self.multi_session[idx]
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
        mri_stack_mono = []
        seg_stack_mono = []
        time_stack_mono = []
        idxes_single = []
        '''
        if len(self.single_session) > self.length_fake_sequence:
            idxes_single = torch.randint(0, len(self.single_session), (self.length_fake_sequence,))
        
        for idx_single in idxes_single:
            data_single = self.single_session[idx_single.item()]
            t_single = data_single[0][2]

            # only keep mono subjects whose age is within the
            # multi-session time range [t₀, inf]
            if t_single < time_stack[0]:
                continue
            session = tio.Subject(
                image=tio.ScalarImage(data_single[0][0]),
                label=tio.LabelMap(data_single[0][1])
                if data_single[0][1] is not None else None,
            )
            if self.transform is not None:
                session = self.transform(session)

            mri_stack_mono.append(session.image.data)
            if session.label is not None:
                seg_stack_mono.append(session.label.data)
            time_stack_mono.append(t_single)
        '''
        all_mri = mri_stack + mri_stack_mono
        all_seg = seg_stack + seg_stack_mono
        all_times = time_stack + time_stack_mono

        # flag: 0 = multi-session (real), 1 = mono-session (fake)
        is_mono = ([0] * len(mri_stack)) + ([1] * len(mri_stack_mono))

        # ── 4. sort by age — multi-session first on tie ────────────────
        # stable sort preserves relative order within same age,
        # and multi-session entries come first in all_times (built first)
        # so a stable sort on age keeps multi before mono at equal ages
        sorted_indices = sorted(
            range(len(all_times)),
            key=lambda i: (all_times[i], is_mono[i]))  # (age, is_mono) → multi first

        all_mri = [all_mri[i] for i in sorted_indices]
        all_seg = [all_seg[i] for i in sorted_indices]
        all_times = [all_times[i] for i in sorted_indices]
        is_mono = [is_mono[i] for i in sorted_indices]

        for i in range(len(all_times) - 1, 0, -1):
            if all_times[i] == all_times[i - 1]:
                # both same time — drop the mono one
                if is_mono[i]:
                    del all_mri[i]
                    del all_seg[i]
                    del all_times[i]
                    del is_mono[i]
                elif is_mono[i - 1]:
                    del all_mri[i - 1]
                    del all_seg[i - 1]
                    del all_times[i - 1]
                    del is_mono[i - 1]
                # if both are mono or both are multi — keep first, drop second
                else:
                    del all_mri[i]
                    del all_seg[i]
                    del all_times[i]
                    del is_mono[i]


        # ── 5. stack ──────────────────────────────────────────────────
        mri_stack_out = torch.stack(all_mri, dim=0)  # (T_total, 1, X, Y, Z)

        if len(all_seg) > 0:
            seg_stack_out = torch.stack(all_seg, dim=0)  # (T_total, C, X, Y, Z)
        else:
            seg_stack_out = torch.empty(0)

        time_stack_out = torch.tensor(all_times, dtype=torch.float)  # (T_total,)
        is_mono_out = torch.tensor(is_mono, dtype=torch.bool)  # (T_total,)
        # is_mono_out[i] = False → real multi-session timepoint
        # is_mono_out[i] = True  → mono-session subject (no NCC supervision)

        return mri_stack_out, seg_stack_out, time_stack_out, is_mono_out




class SpatioTemporalDatasetValidation(torch.utils.data.dataset.Dataset):
    def __init__(self, data, transform=None):
        '''
        PairwiseSubjectsDataset
        :param subjects: Sequence of subjects
        :param transform: Transform composition applying to the subjects
        '''
        super().__init__()
        self.transform = transform
        self.discriminator_phase = False
        self.multi_session = []
        for i in range(len(data)):
            if len(data[i]) > 1:
                self.multi_session.append(data[i])
            # IGNORE >3 (val)


    def __len__(self):
        '''
            Return the number of subjects in the dataset
        '''
        return len(self.multi_session)

    def get_subject_affine(self, idx: int) -> torch.Tensor:
        data = self.multi_session[idx]
        session = tio.Subject(
            image=tio.ScalarImage(data[0][0]),
            label=tio.LabelMap(data[0][1]) if data[0][1] is not None else None,
        )
        return session['image'].data.affine

    def __getitem__(self, idx: int) -> tio.Subject:
        '''
            Get the sequence at index idx
            :param idx: index of the sequence
        '''

        mri_stack = []
        seg_stack = []
        time_stack = []
        data = self.multi_session[idx]
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

        # ── 5. stack ──────────────────────────────────────────────────
        #mri_stack.reverse()
        #seg_stack.reverse()
        #time_stack.reverse()
        mri_stack_out = torch.stack(mri_stack, dim=0)  # (T_total, 1, X, Y, Z)

        if len(seg_stack) > 0:
            seg_stack_out = torch.stack(seg_stack, dim=0)  # (T_total, 1, X, Y, Z)
        else:
            seg_stack_out = torch.empty(0)

        time_stack_out = torch.tensor(time_stack, dtype=torch.float)  # (T_total,)
        # is_mono_out[i] = False → real multi-session timepoint
        # is_mono_out[i] = True  → mono-session subject (no NCC supervision)

        return mri_stack_out, seg_stack_out, time_stack_out
