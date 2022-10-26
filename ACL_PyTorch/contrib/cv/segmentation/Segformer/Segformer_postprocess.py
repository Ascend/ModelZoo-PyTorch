# Copyright 2022 Huawei Technologies Co., Ltd
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

import numpy as np
import torch
import argparse
import os
from mmseg.core.evaluation import metrics
from PIL import Image
import json
from tqdm import tqdm



def read_info_from_json(json_path):
    '''
    input: sumary.json path
    output: a dict read from sumary.json
    '''
    if os.path.exists(json_path) is False:
        exit(json_path + ' is not exist')
    with open(json_path, 'r') as f:
        load_data = json.load(f)
        file_info = load_data['filesinfo']
        return file_info


class IntersectAndUnion(object):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map : Mapping old labels to new labels.
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

    Returns:
        IoU
        Acc
    """

    def __init__(self, num_classes, ignore_index, label_map, reduce_zero_label=False):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.label_map = label_map
        self.reduce_zero_label = reduce_zero_label
        self.total_area_intersect = torch.zeros((num_classes,), dtype=torch.float64)
        self.total_area_union = torch.zeros((num_classes,), dtype=torch.float64)
        self.total_area_pred_label = torch.zeros((num_classes,), dtype=torch.float64)
        self.total_area_label = torch.zeros((num_classes,), dtype=torch.float64)

    def update(self, output, gt_seg_map):
        """ update  """
        [area_intersect, area_union, area_pred_label, area_label] = \
            metrics.intersect_and_union(
                output, gt_seg_map, self.num_classes, self.ignore_index,
                self.label_map, self.reduce_zero_label)
        self.total_area_intersect += area_intersect.to(torch.float64)
        self.total_area_union += area_union.to(torch.float64)
        self.total_area_pred_label += area_pred_label.to(torch.float64)
        self.total_area_label += area_label.to(torch.float64)

    def get(self):
        """ get result """
        iou = self.total_area_intersect / self.total_area_union
        acc = self.total_area_intersect / self.total_area_label
        all_acc = self.total_area_intersect.sum() / self.total_area_label.sum()
        mIoU = np.round(np.nanmean(iou) * 100, 2)
        aAcc = np.round(np.nanmean(all_acc) * 100, 2)
        return {'aAcc': aAcc, 'mIoU': mIoU}


def eval_metrics(_json_files_info,
                 _dataset_path,
                 _result_path,
                 _ori_suffix='_leftImg8bit.bin',
                 _gt_suffix='_gtFine_labelTrainIds.png',
                 num_classes=19,
                 ignore_index=255,
                 label_map=None,
                 reduce_zero_label=False):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    # initial metric
    label_map = dict()
    metric = IntersectAndUnion(num_classes, ignore_index, label_map, reduce_zero_label)

    with tqdm(total=500) as pbar:
        for idx in _json_files_info:
            pbar.update(1)
            input_path = _json_files_info[idx]['infiles'][0]
            outfile = _json_files_info[idx]['outfiles'][0]
            seg_map_path = str(input_path).replace(_ori_suffix, _gt_suffix)
            seg_map_path = seg_map_path.replace('prep_dataset', 'cityscapes/gtFine/val/')
            seg_map_path = seg_map_path.replace("frankfurt", "frankfurt/frankfurt")
            seg_map_path = seg_map_path.replace("lindau", "lindau/lindau")
            seg_map_path = seg_map_path.replace("munster", "munster/munster")
            seg_map_path = seg_map_path.replace("./", _dataset_path)
            if seg_map_path is not None:
                seg_map = Image.open(seg_map_path)
                seg_map = np.array(seg_map, dtype=np.uint8)

                output = np.fromfile(outfile, dtype=np.uint32).reshape(1024, 2048)
                output = output.astype(np.uint8)
                metric.update(output, seg_map)
            else:
                exit("[ERROR] " + seg_map_path + " not found, please check the file")

    # get result
    result = metric.get()
    print(result)
    with open(_result_path + '.txt', 'w') as f:
        f.write('aAcc: {}\n'.format(result['aAcc']))
        f.write('mIoU: {}\n'.format(result['mIoU']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('mIoU calculate')
    parser.add_argument('--json_path',
                        help='path to om sumary.json file')
    parser.add_argument('--dataset_path',
                        help='path to the dataset')
    parser.add_argument('--result_path', default="./postprocess_result",
                        help='path to save the script result, default ./postprocess_result.txt')

    args = parser.parse_args()

    json_path = args.json_path
    result_path = args.result_path
    dataset_path = args.dataset_path
    print("json_path :", json_path)
    json_files_info = read_info_from_json(json_path)
    eval_metrics(_json_files_info = json_files_info, _result_path = result_path, _dataset_path = dataset_path)
