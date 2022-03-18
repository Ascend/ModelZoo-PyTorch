# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from glob import glob
from tqdm import trange
from multiprocessing import cpu_count, Pool
import torchbiomed.utils as utils
import SimpleITK as sitk


def normalize_lung_CT(read_path, save_path, uids):
    MIN_BOUND = -1000
    MAX_BOUND = 400
    img_spacing = (vox_spacing, vox_spacing, vox_spacing)

    for i in trange(len(uids)):
        uid = uids[i]
        img_path = os.path.join(read_path, uid + '.mhd')
        itk_img = sitk.ReadImage(img_path)
        (x_space, y_space, z_space) = itk_img.GetSpacing()
        spacing_old = (z_space, y_space, x_space)
        img_array = sitk.GetArrayFromImage(itk_img)
        img, _, _ = utils.resample_volume(img_array,
                                          spacing_old,
                                          img_spacing,
                                          bounds=(MIN_BOUND, MAX_BOUND))
        utils.save_updated_image(img, os.path.join(save_path, uid + '.mhd'),
                                 itk_img.GetOrigin(), img_spacing)


def normalize_lung_mask(read_path, save_path, uids):
    img_spacing = (vox_spacing, vox_spacing, vox_spacing)
    for i in trange(len(uids)):
        uid = uids[i]
        img_path = os.path.join(read_path, uid + '.mhd')
        itk_img = sitk.ReadImage(img_path)
        (x_space, y_space, z_space) = itk_img.GetSpacing()
        spacing_old = (z_space, y_space, x_space)
        img_array = sitk.GetArrayFromImage(itk_img)
        img, _, _ = utils.resample_volume(img_array, spacing_old, img_spacing)
        img[img < 1] = 0
        img[img > 0] = 1
        utils.save_updated_image(img, os.path.join(save_path, uid + '.mhd'),
                                 itk_img.GetOrigin(), img_spacing, useCompression=True)


if __name__ == '__main__':
    root_path = sys.argv[1]
    vox_spacing = float(sys.argv[2])
    Z_MAX = int(sys.argv[3])
    Y_MAX = int(sys.argv[4])
    X_MAX = int(sys.argv[5])
    print('Normalize with spacing %f'%vox_spacing)
    print('Shape images to %dx%dx%d'%(Z_MAX,Y_MAX,X_MAX))
    lung_ct_path = os.path.join(root_path, 'lung_ct_image')
    lung_mask_path = os.path.join(root_path, 'seg-lungs-LUNA16')
    lung_ct_save_path = os.path.join(root_path, 'normalized_lung_ct')
    lung_mask_save_path = os.path.join(root_path, 'normalized_lung_mask')
    if not os.path.isdir(lung_mask_save_path):
        os.mkdir(lung_mask_save_path)
    if not os.path.isdir(lung_ct_save_path):
        os.mkdir(lung_ct_save_path)

    utils.init_dims3D(Z_MAX, Y_MAX, X_MAX, vox_spacing)
    uids = glob(os.path.join(lung_ct_path, '*.mhd'))
    uids = [os.path.basename(uid)[:-4] for uid in uids]
    print('Find %d cases.' % len(uids))

    core_num = cpu_count()
    if core_num > 8: core_num = 8
    sub_len = len(uids) // core_num
    if len(uids) % core_num != 0:
        sub_len += 1
        
    print('Normalizing lung ct...')
    p0 = Pool(core_num)
    for i in range(core_num):
        start = i * sub_len
        end = start + sub_len if start + sub_len < len(uids) else len(uids)
        p0.apply_async(
            normalize_lung_CT,
            args=(lung_ct_path, lung_ct_save_path, uids[start:end]),
        )
    p0.close()
    p0.join()

    print('Normalizing lung mask...')
    p1 = Pool(core_num)
    for i in range(core_num):
        start = i * sub_len
        end = start + sub_len if start + sub_len < len(uids) else len(uids)
        p1.apply_async(
            normalize_lung_mask,
            args=(lung_mask_path, lung_mask_save_path, uids[start:end]),
        )
    p1.close()
    p1.join()

    print('Dataset normalization done.')
