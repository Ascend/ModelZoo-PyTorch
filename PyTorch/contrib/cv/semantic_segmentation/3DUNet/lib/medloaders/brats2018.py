# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

import lib.augment3D as augment3D
import lib.utils as utils
from lib.medloaders import medical_image_process as img_loader
from lib.medloaders.medical_loader_utils import create_sub_volumes


class MICCAIBraTS2018(Dataset):
    """
    Code for reading the infant brain MICCAIBraTS2018 challenge
    """

    def __init__(self, args, mode, dataset_path='./datasets', classes=5, crop_dim=(32, 32, 32), split_idx=10,
                 samples=10,
                 load=False):
        """
        :param mode: 'train','val','test'
        :param dataset_path: root dataset folder
        :param crop_dim: subvolume tuple
        :param split_idx: 1 to 10 values
        :param samples: number of sub-volumes that you want to create
        """
        self.mode = mode
        self.root = str(dataset_path)
        self.training_path = self.root + '/MICCAI_BraTS_2018_Data_Training/'
        self.testing_path = self.root + ' '
        self.CLASSES = 4
        self.full_vol_dim = (240, 240, 155)  # slice, width, height
        self.crop_size = crop_dim
        self.threshold = args.threshold
        self.normalization = args.normalization
        self.augmentation = args.augmentation
        self.list = []
        self.samples = samples
        self.full_volume = None
        self.classes = classes
        self.save_name = self.root + '/MICCAI_BraTS_2018_Data_Training/brats2018-list-' + mode + '-samples-' + str(
            samples) + '.txt'

        if self.augmentation:
            if mode == "train":
                # self.transform = augment3D.RandomChoice(
                #     transforms=[augment3D.GaussianNoise(mean=0, std=0.01), augment3D.RandomFlip(),
                #                 augment3D.ElasticTransform()], p=0.5)
                self.transform = augment3D.ComposeTransforms(
                    transforms=[augment3D.GaussianNoise(mean=0, std=0.01), augment3D.RandomFlip(),
                                augment3D.ElasticTransform()], p=1.0)

        
        
        try:
            ## load pre-generated data
            list_IDsT1 = sorted(glob.glob(os.path.join(self.training_path, '*GG/*/*t1.nii.gz')))
            self.affine = img_loader.load_affine_matrix(list_IDsT1[0])
            self.list = utils.load_list(self.save_name)
            print("utils.load_list(self.save_name) ok.")
            return
        except: 
            print("utils.load_list(self.save_name) failed. generating......")
            pass

        subvol = '_vol_' + str(crop_dim[0]) + 'x' + str(crop_dim[1]) + 'x' + str(crop_dim[2])
        self.sub_vol_path = self.root + '/MICCAI_BraTS_2018_Data_Training/generated/' + mode + subvol + '/'
        utils.make_dirs(self.sub_vol_path)

        list_IDsT1 = sorted(glob.glob(os.path.join(self.training_path, '*GG/*/*t1.nii.gz')))
        list_IDsT1ce = sorted(glob.glob(os.path.join(self.training_path, '*GG/*/*t1ce.nii.gz')))
        list_IDsT2 = sorted(glob.glob(os.path.join(self.training_path, '*GG/*/*t2.nii.gz')))
        list_IDsFlair = sorted(glob.glob(os.path.join(self.training_path, '*GG/*/*_flair.nii.gz')))
        labels = sorted(glob.glob(os.path.join(self.training_path, '*GG/*/*_seg.nii.gz')))
        # print(len(list_IDsT1),len(list_IDsT2),len(list_IDsFlair),len(labels))

        self.affine = img_loader.load_affine_matrix(list_IDsT1[0])

        if self.mode == 'train':
            list_IDsT1 = list_IDsT1[:split_idx]
            list_IDsT1ce = list_IDsT1ce[:split_idx]
            list_IDsT2 = list_IDsT2[:split_idx]
            list_IDsFlair = list_IDsFlair[:split_idx]
            labels = labels[:split_idx]

            self.list = create_sub_volumes(list_IDsT1, list_IDsT1ce, list_IDsT2, list_IDsFlair, labels,
                                           dataset_name="brats2018", mode=mode, samples=samples,
                                           full_vol_dim=self.full_vol_dim, crop_size=self.crop_size,
                                           sub_vol_path=self.sub_vol_path, normalization=self.normalization,
                                           th_percent=self.threshold)
        elif self.mode == 'val':
            list_IDsT1 = list_IDsT1[split_idx:]
            list_IDsT1ce = list_IDsT1ce[split_idx:]
            list_IDsT2 = list_IDsT2[split_idx:]
            list_IDsFlair = list_IDsFlair[split_idx:]
            labels = labels[split_idx:]
            self.list = create_sub_volumes(list_IDsT1, list_IDsT1ce, list_IDsT2, list_IDsFlair, labels,
                                           dataset_name="brats2018", mode=mode, samples=samples,
                                           full_vol_dim=self.full_vol_dim, crop_size=self.crop_size,
                                           sub_vol_path=self.sub_vol_path, normalization=self.normalization,
                                           th_percent=self.threshold)

        elif self.mode == 'test':
            self.list_IDsT1 = sorted(glob.glob(os.path.join(self.testing_path, '*GG/*/*t1.nii.gz')))
            self.list_IDsT1ce = sorted(glob.glob(os.path.join(self.testing_path, '*GG/*/*t1ce.nii.gz')))
            self.list_IDsT2 = sorted(glob.glob(os.path.join(self.testing_path, '*GG/*/*t2.nii.gz')))
            self.list_IDsFlair = sorted(glob.glob(os.path.join(self.testing_path, '*GG/*/*_flair.nii.gz')))
            self.labels = None

        utils.save_list(self.save_name, self.list)
        print("utils.save_list(self.save_name, self.list) ok.")

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        f_t1, f_t1ce, f_t2, f_flair, f_seg = self.list[index]
        img_t1, img_t1ce, img_t2, img_flair, img_seg = np.load(f_t1), np.load(f_t1ce), np.load(f_t2), np.load(
            f_flair), np.load(f_seg)
        if self.mode == 'train' and self.augmentation:
            [img_t1, img_t1ce, img_t2, img_flair], img_seg = self.transform([img_t1, img_t1ce, img_t2, img_flair],
                                                                            img_seg)

            return torch.FloatTensor(img_t1.copy()).unsqueeze(0), torch.FloatTensor(img_t1ce.copy()).unsqueeze(
                0), torch.FloatTensor(img_t2.copy()).unsqueeze(0), torch.FloatTensor(img_flair.copy()).unsqueeze(
                0), torch.FloatTensor(img_seg.copy())

        return img_t1, img_t1ce, img_t2, img_flair, img_seg
