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
import datetime
import yaml
import shutil
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import struct
from PIL import Image
from torchvision import transforms
from utils import SegmentationMetric, get_color_pallete


class Evaluator(object):
    def __init__(self, cityscapes_path, folder_davinci_target, outdir):

        loc = "cpu"
        self.device = torch.device(loc)
        print("===device===:",self.device)
        # get valid dataset images and targets
        self.image_paths, self.mask_paths = _get_city_pairs(cityscapes_path, "val")

        self.annotation_file_path = folder_davinci_target

        self.outdir = outdir

        # evaluation metrics
        self.metric = SegmentationMetric(19)

    def eval(self):
        self.metric.reset()
        
        print("Start validation, Total sample: {:d}".format(len(self.image_paths)))
        list_time = []
        lsit_pixAcc = []
        list_mIoU = []

        for i in range(len(self.image_paths)):
            filename = os.path.basename(self.image_paths[i])
            annotation_file = os.path.join(self.annotation_file_path, filename.split('.')[0])
            
            mask = Image.open(self.mask_paths[i])  # mask shape: (W,H)    
            mask = self._mask_transform(mask)  # mask shape: (H,w)
            mask = mask.to(self.device)

            with torch.no_grad():
                start_time = time.time()
                outputs = self.file2tensor(annotation_file).to(self.device)
                end_time = time.time()
                step_time = end_time - start_time
            self.metric.update(outputs, mask)
            pixAcc, mIoU = self.metric.get()
            list_time.append(step_time)
            lsit_pixAcc.append(pixAcc)
            list_mIoU.append(mIoU)
            print("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}, time: {:.3f}s".format(
                i + 1, pixAcc * 100, mIoU * 100, step_time))
            
            '''
            filename = os.path.basename(self.image_paths[i])
            prefix = filename.split('.')[0]
         
            # save pred 
            pred = torch.argmax(outputs, 1)
            pred = pred.cpu().data.numpy()
            pred = pred.squeeze(0)
            pred = get_color_pallete(pred, "citys")
            pred.save(os.path.join(outdir, prefix + "_mIoU_{:.3f}.png".format(mIoU)))

            # save image
            image = Image.open(self.image_paths[i]).convert('RGB')  # image shape: (W,H,3)
            image.save(os.path.join(outdir, prefix + '_src.png'))

            # save target
            mask = Image.open(self.mask_paths[i])  # mask shape: (W,H)
            mask = self._class_to_index(np.array(mask).astype('int32'))
            mask = get_color_pallete(mask, "citys")
            mask.save(os.path.join(outdir, prefix + '_label.png'))
            '''
        average_pixAcc = sum(lsit_pixAcc) / len(lsit_pixAcc)
        average_mIoU = sum(list_mIoU) / len(list_mIoU)
        average_time = sum(list_time) / len(list_time)
        self.current_mIoU = average_mIoU
        print("Evaluate: Average mIoU: {:.3f}, Average pixAcc: {:.3f}, Average time: {:.3f}"
              .format(average_mIoU, average_pixAcc, average_time))

    def _img_transform(self, image):
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        image = image_transform(image)
        return image

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
    
        dim_res = np.array(res).reshape(1,19,1024,2048)
        tensor_res = torch.tensor(dim_res, dtype=torch.float32)
        print(filepath, tensor_res.dtype, tensor_res.shape)

        return tensor_res


def _get_city_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith('.png'):
                    """
                    For example:
                        root = "./Cityscapes/leftImg8bit/train/aachen"
                        filename = "aachen_xxx_leftImg8bit.png"
                        imgpath = "./Cityscapes/leftImg8bit/train/aachen/aachen_xxx_leftImg8bit.png"
                        foldername = "aachen"
                        maskname = "aachen_xxx_gtFine_labelIds.png"
                        maskpath = "./Cityscapes/gtFine/train/aachen/aachen_xxx_gtFine_labelIds"
                    """
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
        # the path to store the results json path
        outdir = sys.argv[3]

    except IndexError:
        print("Stopped!")
        exit(1)

    if not (os.path.exists(cityscapes_path)):
        print("config file folder does not exist.")
    if not (os.path.exists(folder_davinci_target)):
        print("target file folder does not exist.")
    if not (os.path.exists(outdir)):
        print("output file folder does not exist.")
        os.makedirs(outdir)

    evaluator = Evaluator(cityscapes_path, folder_davinci_target, outdir)
    evaluator.eval()
