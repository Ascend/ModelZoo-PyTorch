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

import sys
import os
import torch
import cv2
from PIL import Image
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import argparse
import torch.nn.functional as F

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", default = "./lmcout")
    parser.add_argument("--label_images", default = "./SegmentationClass/")
    parser.add_argument("--labels", default = "./val.txt")
    flags = parser.parse_args()
    npu_dir = flags.result_path
    mask_img_dir = flags.label_images
    input_list = flags.labels
    metrics = StreamSegMetrics(21)
    metrics.reset()
    with open(os.path.join(input_list), "r") as f:
        file_names = [x.strip() for x in f.readlines()]
    
    npu_images = [os.path.join(npu_dir, x + "_0.bin") for x in file_names]
    mask_images = [os.path.join(mask_img_dir, x + ".png") for x in file_names]
    total = len(npu_images)
    for i in range(total):
        print("npu_images:", npu_images[i])
        pred = np.fromfile(npu_images[i], dtype='float16').reshape((1,21,513,513)).astype(np.float32)
        pred = torch.from_numpy(pred)
        pred = pred.max(dim=1)[1].numpy()
        
        print("pred.shape:", pred.shape)
        
        target = Image.open(mask_images[i])
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        val_transformer = transforms.Compose([
            transforms.Resize(513),
            transforms.CenterCrop(513)
        ])
        
        target = val_transformer(target)
        target = torch.from_numpy( np.array( target, dtype="uint8") )
        target = target.cpu().numpy().reshape(1,513,513)
        metrics.update(target, pred)
    score = metrics.get_results()
    print(score)
