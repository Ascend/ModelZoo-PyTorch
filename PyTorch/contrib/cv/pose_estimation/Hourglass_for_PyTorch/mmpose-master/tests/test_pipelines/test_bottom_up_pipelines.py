# -*- coding: utf-8 -*-
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

import copy
import os.path as osp

import numpy as np
import pytest
import xtcocotools
from xtcocotools.coco import COCO

from mmpose.datasets.pipelines import (BottomUpGenerateTarget,
                                       BottomUpGetImgSize,
                                       BottomUpRandomAffine,
                                       BottomUpRandomFlip, BottomUpResizeAlign,
                                       LoadImageFromFile)


def _get_mask(coco, anno, img_id):
    img_info = coco.loadImgs(img_id)[0]

    m = np.zeros((img_info['height'], img_info['width']), dtype=np.float32)

    for obj in anno:
        if obj['iscrowd']:
            rle = xtcocotools.mask.frPyObjects(obj['segmentation'],
                                               img_info['height'],
                                               img_info['width'])
            m += xtcocotools.mask.decode(rle)
        elif obj['num_keypoints'] == 0:
            rles = xtcocotools.mask.frPyObjects(obj['segmentation'],
                                                img_info['height'],
                                                img_info['width'])
            for rle in rles:
                m += xtcocotools.mask.decode(rle)

    return m < 0.5


def _get_joints(anno, ann_info, int_sigma):
    num_people = len(anno)

    if ann_info['scale_aware_sigma']:
        joints = np.zeros((num_people, ann_info['num_joints'], 4),
                          dtype=np.float32)
    else:
        joints = np.zeros((num_people, ann_info['num_joints'], 3),
                          dtype=np.float32)

    for i, obj in enumerate(anno):
        joints[i, :ann_info['num_joints'], :3] = \
            np.array(obj['keypoints']).reshape([-1, 3])
        if ann_info['scale_aware_sigma']:
            # get person box
            box = obj['bbox']
            size = max(box[2], box[3])
            sigma = size / 256 * 2
            if int_sigma:
                sigma = int(np.ceil(sigma))
            assert sigma > 0, sigma
            joints[i, :, 3] = sigma

    return joints


def _check_flip(origin_imgs, result_imgs):
    """Check if the origin_imgs are flipped correctly."""
    h, w, c = origin_imgs.shape
    for i in range(h):
        for j in range(w):
            for k in range(c):
                if result_imgs[i, j, k] != origin_imgs[i, w - 1 - j, k]:
                    return False
    return True


def test_bottomup_pipeline():

    data_prefix = 'tests/data/coco/'
    ann_file = osp.join(data_prefix, 'test_coco.json')
    coco = COCO(ann_file)

    ann_info = {}
    ann_info['flip_pairs'] = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                              [11, 12], [13, 14], [15, 16]]
    ann_info['flip_index'] = [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
    ]

    ann_info['use_different_joint_weights'] = False
    ann_info['joint_weights'] = np.array([
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        1.5
    ],
                                         dtype=np.float32).reshape((17, 1))
    ann_info['image_size'] = np.array(512)
    ann_info['heatmap_size'] = np.array([128, 256])
    ann_info['num_joints'] = 17
    ann_info['num_scales'] = 2
    ann_info['scale_aware_sigma'] = False

    ann_ids = coco.getAnnIds(785)
    anno = coco.loadAnns(ann_ids)
    mask = _get_mask(coco, anno, 785)

    anno = [
        obj for obj in anno if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0
    ]
    joints = _get_joints(anno, ann_info, False)

    mask_list = [mask.copy() for _ in range(ann_info['num_scales'])]
    joints_list = [joints.copy() for _ in range(ann_info['num_scales'])]

    results = {}
    results['dataset'] = 'coco'
    results['image_file'] = osp.join(data_prefix, '000000000785.jpg')
    results['mask'] = mask_list
    results['joints'] = joints_list
    results['ann_info'] = ann_info

    transform = LoadImageFromFile()
    results = transform(copy.deepcopy(results))
    assert results['img'].shape == (425, 640, 3)

    # test HorizontalFlip
    random_horizontal_flip = BottomUpRandomFlip(flip_prob=1.)
    results_horizontal_flip = random_horizontal_flip(copy.deepcopy(results))
    assert _check_flip(results['img'], results_horizontal_flip['img'])

    random_horizontal_flip = BottomUpRandomFlip(flip_prob=0.)
    results_horizontal_flip = random_horizontal_flip(copy.deepcopy(results))
    assert (results['img'] == results_horizontal_flip['img']).all()

    results_copy = copy.deepcopy(results)
    results_copy['mask'] = mask_list[0]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_horizontal_flip(
            copy.deepcopy(results_copy))

    results_copy = copy.deepcopy(results)
    results_copy['joints'] = joints_list[0]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_horizontal_flip(
            copy.deepcopy(results_copy))

    results_copy = copy.deepcopy(results)
    results_copy['joints'] = joints_list[:1]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_horizontal_flip(
            copy.deepcopy(results_copy))

    results_copy = copy.deepcopy(results)
    results_copy['mask'] = mask_list[:1]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_horizontal_flip(
            copy.deepcopy(results_copy))

    # test TopDownAffine
    random_affine_transform = BottomUpRandomAffine(30, [0.75, 1.5], 'short', 0)
    results_affine_transform = random_affine_transform(copy.deepcopy(results))
    assert results_affine_transform['img'].shape == (512, 512, 3)

    random_affine_transform = BottomUpRandomAffine(30, [0.75, 1.5], 'short',
                                                   40)
    results_affine_transform = random_affine_transform(copy.deepcopy(results))
    assert results_affine_transform['img'].shape == (512, 512, 3)

    results_copy = copy.deepcopy(results)
    results_copy['ann_info']['scale_aware_sigma'] = True
    joints = _get_joints(anno, results_copy['ann_info'], False)
    results_copy['joints'] = \
        [joints.copy() for _ in range(results_copy['ann_info']['num_scales'])]
    results_affine_transform = random_affine_transform(results_copy)
    assert results_affine_transform['img'].shape == (512, 512, 3)

    results_copy = copy.deepcopy(results)
    results_copy['mask'] = mask_list[0]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_affine_transform(
            copy.deepcopy(results_copy))

    results_copy = copy.deepcopy(results)
    results_copy['joints'] = joints_list[0]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_affine_transform(
            copy.deepcopy(results_copy))

    results_copy = copy.deepcopy(results)
    results_copy['joints'] = joints_list[:1]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_affine_transform(
            copy.deepcopy(results_copy))

    results_copy = copy.deepcopy(results)
    results_copy['mask'] = mask_list[:1]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_affine_transform(
            copy.deepcopy(results_copy))

    random_affine_transform = BottomUpRandomAffine(30, [0.75, 1.5], 'long', 40)
    results_affine_transform = random_affine_transform(copy.deepcopy(results))
    assert results_affine_transform['img'].shape == (512, 512, 3)

    with pytest.raises(ValueError):
        random_affine_transform = BottomUpRandomAffine(30, [0.75, 1.5],
                                                       'short-long', 40)
        results_affine_transform = random_affine_transform(
            copy.deepcopy(results))

    # test BottomUpGenerateTarget
    generate_multi_target = BottomUpGenerateTarget(2, 30)
    results_generate_multi_target = generate_multi_target(
        copy.deepcopy(results))
    assert 'targets' in results_generate_multi_target
    assert len(results_generate_multi_target['targets']
               ) == results['ann_info']['num_scales']

    # test BottomUpGetImgSize when W > H
    get_multi_scale_size = BottomUpGetImgSize([1])
    results_get_multi_scale_size = get_multi_scale_size(copy.deepcopy(results))
    assert 'test_scale_factor' in results_get_multi_scale_size['ann_info']
    assert 'base_size' in results_get_multi_scale_size['ann_info']
    assert 'center' in results_get_multi_scale_size['ann_info']
    assert 'scale' in results_get_multi_scale_size['ann_info']
    assert results_get_multi_scale_size['ann_info']['base_size'][1] == 512

    # test BottomUpResizeAlign
    transforms = [
        dict(type='ToTensor'),
        dict(
            type='NormalizeTensor',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ]
    resize_align_multi_scale = BottomUpResizeAlign(transforms=transforms)
    results_copy = copy.deepcopy(results_get_multi_scale_size)
    results_resize_align_multi_scale = resize_align_multi_scale(results_copy)
    assert 'aug_data' in results_resize_align_multi_scale['ann_info']

    # test BottomUpGetImgSize when W < H
    results_copy = copy.deepcopy(results)
    results_copy['img'] = np.random.rand(640, 425, 3)
    results_get_multi_scale_size = get_multi_scale_size(results_copy)
    assert results_get_multi_scale_size['ann_info']['base_size'][0] == 512
