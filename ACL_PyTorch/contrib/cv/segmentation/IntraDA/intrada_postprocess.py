# Copyright 2021 Huawei Technologies Co., Ltd
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
import time
import sys
import torch
import numpy as np
import torch.nn as nn
import struct
from PIL import Image

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    

class Evaluator(object):
    def __init__(self, cityscapes_path, folder_davinci_target):

        loc = "cpu"
        self.device = torch.device(loc)
        print("===device===:",self.device)
        # get valid dataset images and targets
        self.image_paths, self.mask_paths = _get_city_pairs(cityscapes_path, "val")

        self.annotation_file_path = folder_davinci_target

    def eval(self):
        
        print("Start validation, Total sample: {:d}".format(len(self.image_paths)))
        list_time = []

        hist = np.zeros((19, 19))
        for i in range(len(self.image_paths)):
            filename = os.path.basename(self.image_paths[i])
            annotation_file = os.path.join(self.annotation_file_path, filename.split('.')[0])
            
            mask = Image.open(self.mask_paths[i])  # mask shape: (W,H)  
            mask = mask.resize((2048,1024),Image.NEAREST)
            mask = self._mask_transform(mask)  # mask shape: (H,w)
            mask = mask.to(self.device)

            with torch.no_grad():
                start_time = time.time()
                outputs = self.file2tensor(annotation_file).to(self.device)
                end_time = time.time()

                outputs_ = outputs.numpy().squeeze().transpose(1, 2, 0)
                outputs_ = np.argmax(outputs_, axis=2)
                hist += fast_hist(mask.cpu().numpy().flatten(), outputs_.flatten(), 19)
                inters_over_union_classes = per_class_iu(hist)
                mIoU = np.nanmean(inters_over_union_classes)
                step_time = end_time - start_time

            list_time.append(step_time)

            print("Sample: {:d}, mIoU: {:.3f}, time: {:.3f}s".format(
                i + 1, mIoU * 100, step_time))
            
        average_time = sum(list_time) / len(list_time)
        print("Evaluate: Average mIoU: {:.3f}, Average time: {:.3f}"
              .format(mIoU * 100, average_time))

    def _mask_transform(self, mask):
        mask = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(mask).astype('int32'))

    def _class_to_index(self, mask):
        # assert the value
        values = np.unique(mask)
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')
        for value in values:
            assert (value in self._mapping)
        
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        
        return self._key[index].reshape(mask.shape)

    def file2tensor(self, annotation_file):
    
        filepath = annotation_file + '_1.bin'
        size = os.path.getsize(filepath)  
        res = []
        L = int(size/4)
        binfile = open(filepath, 'rb')  
        for i in range(L):
            data = binfile.read(4)
            num = struct.unpack('f', data)
            res.append(num[0])
        binfile.close()

        dim_res = np.array(res).reshape(1,19,65,129)
        tensor_res = torch.tensor(dim_res, dtype=torch.float32)
        interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)
        tensor_res = interp(tensor_res)
        print(filepath, tensor_res.dtype, tensor_res.shape)

        return tensor_res


def _get_city_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith('.png'):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split in ('train', 'val'):
        # "./Cityscapes/leftImg8bit/train" or "./Cityscapes/leftImg8bit/val"
        img_folder = os.path.join(folder, 'leftImg8bit/' + split)
        # "./Cityscapes/gtFine/train" or "./Cityscapes/gtFine/val"
        mask_folder = os.path.join(folder, 'gtFine/' + split)
        # img_paths mask_paths
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    return img_paths, mask_paths


if __name__ == '__main__':

    try:
        # dataset file path
        cityscapes_path = sys.argv[1]
        # txt file path
        folder_davinci_target = sys.argv[2]

    except IndexError:
        print("Stopped!")
        exit(1)

    if not (os.path.exists(cityscapes_path)):
        print("config file folder does not exist.")
    if not (os.path.exists(folder_davinci_target)):
        print("target file folder does not exist.")

    evaluator = Evaluator(cityscapes_path, folder_davinci_target)
    evaluator.eval()
