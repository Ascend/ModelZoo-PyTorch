# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================
from data import COCODetection, get_label_map
from utils.augmentations import BaseTransform
from data import cfg, set_cfg
import numpy as np
from torch.autograd import Variable
import argparse
import random
import os
from collections import defaultdict

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--valid_images', default='/home/data/coco/images/', help='the path of validation images')
    parser.add_argument('--valid_annotations', default='/home/data/coco/annotations/instances_val2017.json', help='the path of validation annotations')
    parser.add_argument('--saved_path', default='./', help='the path of binary data and info file')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.set_defaults(display=False, resume=False, shuffle=False,
                        no_sort=False)

    global args
    args = parser.parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})

def prep_coco_cats():
    """ Prepare inverted table for category id lookup given a coco cats object. """
    for coco_cat_id, transformed_cat_id_p1 in get_label_map().items():
        transformed_cat_id = transformed_cat_id_p1 - 1
        coco_cats[transformed_cat_id] = coco_cat_id
        coco_cats_inv[coco_cat_id] = transformed_cat_id

def badhash(x):
    """
    Just a quick and dirty hash function for doing a deterministic shuffle based on image_id.

    Source:
    https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    """
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x =  ((x >> 16) ^ x) & 0xFFFFFFFF
    return x

def preprocess(dataset, save_path = None):
    dataset_size = len(dataset) if args.max_images < 0 else min(args.max_images, len(dataset))
    print("dataset size is : ", dataset_size)
    print()

    # For each class and iou, stores tuples (score, isPositive)
    # Index ap_data[type][iouIdx][classIdx]
    dataset_indices = list(range(len(dataset)))
    print("dataset indices size is :", len(dataset_indices))
    if args.shuffle:
        random.shuffle(dataset_indices)
    elif not args.no_sort:
        hashed = [badhash(x) for x in dataset.ids]
        dataset_indices.sort(key=lambda x: hashed[x])

    dataset_indices = dataset_indices[:dataset_size]

    if save_path is None:
        save_path = os.path.join(args.saved_path, 'prep_dataset')

    if os.path.exists(save_path) == False:
        os.mkdir(save_path)
    else:
        print('dir exist!')

    # Main eval loop
    with open(os.path.join(args.saved_path, 'yolact_prep_bin.info'), 'w+') as f:
        for it, image_idx in enumerate(dataset_indices):
            img, gt, gt_masks, h, w, num_crowd = dataset.pull_item(image_idx)
            # Test flag, do not upvote
            batch = Variable(img.unsqueeze(0))
            batch_numpy = np.array(batch).astype(np.float32)

            binFileName = os.path.join(save_path, 'coco_val2017_' + str(image_idx) + '.bin')

            batch_numpy.tofile(binFileName)

            line = str(it) + ' ' + binFileName + ' ' + '550 550\n'
            f.write(line)
            if it % 100 == 0:
                print('[INFO][PreProcess]', 'CurSampleNum:', it)

if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        set_cfg(args.config)

    else:
        args.config = 'yolact_base_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    if not os.path.exists('results'):
        os.makedirs('results')

    #if args.image is None and args.video is None and args.images is None:
    dataset = COCODetection(args.valid_images, args.valid_annotations,
                            transform=BaseTransform(), has_gt=cfg.dataset.has_gt)
    prep_coco_cats()

    preprocess(dataset)


