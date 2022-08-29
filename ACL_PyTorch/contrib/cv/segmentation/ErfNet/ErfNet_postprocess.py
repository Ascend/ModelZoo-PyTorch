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
import json
from PIL import Image
from torchvision.transforms import Compose, Resize
import numpy as np
import time
import torch
from tqdm import tqdm


class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert isinstance(tensor, torch.LongTensor) or isinstance(tensor,
                                                                  torch.ByteTensor), 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


labels_transform = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),  # ignore label to 19
])


class IouEval:

    def __init__(self, nClasses, ignoreIndex=19):
        self.nClasses = nClasses
        self.ignoreIndex = ignoreIndex if nClasses > ignoreIndex else -1  # if ignoreIndex is larger than nClasses, consider no ignoreIndex
        self.reset()

    def reset(self):
        classes = self.nClasses if self.ignoreIndex == -1 else self.nClasses - 1
        self.tp = torch.zeros(classes).double()
        self.fp = torch.zeros(classes).double()
        self.fn = torch.zeros(classes).double()

    def addBatch(self, x, y):  # x=preds, y=targets
        # sizes should be "batch_size x nClasses x H x W"

        # if size is "batch_size x 1 x H x W" scatter to onehot
        if (x.size(1) == 1):
            x_onehot = torch.zeros(x.size(0), self.nClasses, x.size(2), x.size(3))
            x_onehot.scatter_(1, x, 1).float()
        else:
            x_onehot = x.float()

        if (y.size(1) == 1):
            y_onehot = torch.zeros(y.size(0), self.nClasses, y.size(2), y.size(3))
            y_onehot.scatter_(1, y, 1).float()
        else:
            y_onehot = y.float()

        if (self.ignoreIndex != -1):
            ignores = y_onehot[:, self.ignoreIndex].unsqueeze(1)
            x_onehot = x_onehot[:, :self.ignoreIndex]
            y_onehot = y_onehot[:, :self.ignoreIndex]
        else:
            ignores = 0

        tpmult = x_onehot * y_onehot  # times prediction and gt coincide is 1
        tp = torch.sum(torch.sum(torch.sum(tpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3,
                       keepdim=True).squeeze()
        fpmult = x_onehot * (
                1 - y_onehot - ignores)  # times prediction says its that class and gt says its not (subtracting cases when its ignore label!)
        fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3,
                       keepdim=True).squeeze()
        fnmult = (1 - x_onehot) * (y_onehot)  # times prediction says its not that class and gt says it is
        fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3,
                       keepdim=True).squeeze()

        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()

    def getIoU(self):
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        return torch.mean(iou), iou  # returns "iou mean", "iou per class"


# Class for colors
class Colors:
    RED = '\033[31;1m'
    GREEN = '\033[32;1m'
    YELLOW = '\033[33;1m'
    BLUE = '\033[34;1m'
    MAGENTA = '\033[35;1m'
    CYAN = '\033[36;1m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'


# Colored value output if colorized flag is activated.
def getColorEntry(val):
    if not isinstance(val, float):
        return Colors.ENDC
    if (val < .20):
        return Colors.RED
    elif (val < .40):
        return Colors.YELLOW
    elif (val < .60):
        return Colors.BLUE
    elif (val < .80):
        return Colors.CYAN
    else:
        return Colors.GREEN


def is_label(filename):
    return filename.endswith("_labelTrainIds.png")


def main():
    iouEvalVal = IouEval(20)
    with open(result_sumary, "r") as f:
        row_data = json.load(f)
        values = list(row_data['filesinfo'].values())
        for i in tqdm(range(len(values))):
            # result = np.fromfile(values[i]['outfiles'][0], dtype='float32')
            result = np.fromfile(values[i]['outfiles'][0], dtype='float16').astype(np.float32)
            result = np.reshape(result, (20, 512, 1024))
            outputs = torch.from_numpy(result)
            outputs = outputs.unsqueeze(0)
            label_name = os.path.basename(values[i]['infiles'][0]).split('.')[0][:-11] + "gtFine_labelTrainIds.png"
            label_path = os.path.join(label_dir, label_name)
            labels = Image.open(label_path).convert("P")
            labels = labels_transform(labels)
            labels = labels.unsqueeze(0)
            final_outputs = outputs.max(1)[1].unsqueeze(1)
            iouEvalVal.addBatch(final_outputs, labels)
    iouVal, iou_classes = iouEvalVal.getIoU()

    print("iou is ", iouVal)
    print("iou classes is ", iou_classes)


if __name__ == "__main__":
    result_sumary = sys.argv[1]  # /home/zbh1/output/2022_07_19-11_53_09/sumary.json
    label_dir = sys.argv[2]  # /home/zbh1/ErfNet/gt_label/
    main()
