# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==========================================================================

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import mmcv

from mmocr.utils import convert_annotations


def collect_files(img_dir, gt_dir, split):
    """Collect all images and their corresponding groundtruth files.

    Args:
        img_dir (str): The image directory
        gt_dir (str): The groundtruth directory

    Returns:
        files (list): The list of tuples (img_file, groundtruth_file)
    """
    assert isinstance(img_dir, str)
    assert img_dir
    assert isinstance(gt_dir, str)
    assert gt_dir

    ann_list, imgs_list, splits = [], [], []
    for img in os.listdir(img_dir):
        img_path = osp.join(img_dir, img)
        imgs_list.append(img_path)
        ann_list.append(osp.join(gt_dir, 'gt_' + img.split('.')[0] + '.txt'))
        splits.append(split)

    files = list(zip(sorted(imgs_list), sorted(ann_list), splits))
    assert len(files), f'No images found in {img_dir}'
    print(f'Loaded {len(files)} images from {img_dir}')

    return files


def collect_annotations(files, nproc=1):
    """Collect the annotation information.

    Args:
        files (list): The list of tuples (image_file, groundtruth_file)
        nproc (int): The number of process to collect annotations

    Returns:
        images (list): The list of image information dicts
    """
    assert isinstance(files, list)
    assert isinstance(nproc, int)

    if nproc > 1:
        images = mmcv.track_parallel_progress(
            load_img_info, files, nproc=nproc)
    else:
        images = mmcv.track_progress(load_img_info, files)

    return images


def load_img_info(files):
    """Load the information of one image.

    Args:
        files (tuple): The tuple of (img_file, groundtruth_file, split)

    Returns:
        img_info (dict): The dict of the img and annotation information
    """
    assert isinstance(files, tuple)

    img_file, gt_file, split = files
    # read imgs while ignoring orientations
    img = mmcv.imread(img_file, 'unchanged')

    img_info = dict(
        file_name=osp.join(osp.basename(img_file)),
        height=img.shape[0],
        width=img.shape[1],
        segm_file=osp.join(osp.basename(gt_file)))

    # IC13 uses different separator in gt files
    if split == 'training':
        separator = ' '
    elif split == 'test':
        separator = ','
    else:
        raise NotImplementedError
    if osp.splitext(gt_file)[1] == '.txt':
        img_info = load_txt_info(gt_file, img_info, separator)
    else:
        raise NotImplementedError

    return img_info


def load_txt_info(gt_file, img_info, separator):
    """Collect the annotation information.

    The annotation format is as the following:
    [train]
    left top right bottom "transcription"
    [test]
    left, top, right, bottom, "transcription"

    Args:
        gt_file (str): The path to ground-truth
        img_info (dict): The dict of the img and annotation information

    Returns:
        img_info (dict): The dict of the img and annotation information
    """
    anno_info = []
    with open(gt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            xmin, ymin, xmax, ymax = line.split(separator)[0:4]
            x = max(0, int(xmin))
            y = max(0, int(ymin))
            w = int(xmax) - x
            h = int(ymax) - y
            bbox = [x, y, w, h]
            segmentation = [x, y, x + w, y, x + w, y + h, x, y + h]

            anno = dict(
                iscrowd=0,
                category_id=1,
                bbox=bbox,
                area=w * h,
                segmentation=[segmentation])
            anno_info.append(anno)
    img_info.update(anno_info=anno_info)

    return img_info


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and test set of IC13')
    parser.add_argument('root_path', help='Root dir path of IC13')
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path

    for split in ['training', 'test']:
        print(f'Processing {split} set...')
        with mmcv.Timer(print_tmpl='It takes {}s to convert IC13 annotation'):
            files = collect_files(
                osp.join(root_path, 'imgs', split),
                osp.join(root_path, 'annotations', split), split)
            image_infos = collect_annotations(files, nproc=args.nproc)
            convert_annotations(
                image_infos, osp.join(root_path,
                                      'instances_' + split + '.json'))


if __name__ == '__main__':
    main()
