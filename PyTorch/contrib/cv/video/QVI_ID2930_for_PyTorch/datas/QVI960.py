# Copyright 2020 Huawei Technologies Co., Ltd
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
# dataloader for multi frames (acceleration), modified from superslomo

import torch.utils.data as data
from PIL import Image
import os
import os.path
import random
import sys

import cv2

def _make_dataset(dir):
    framesPath = []
    # Find and loop over all the clips in root `dir`.
    for index, folder in enumerate(os.listdir(dir)):
        clipsFolderPath = os.path.join(dir, folder)
        # Skip items which are not folders.
        if not (os.path.isdir(clipsFolderPath)):
            continue
        framesPath.append([])
        # Find and loop over all the frames inside the clip.
        # for image in sorted(os.listdir(clipsFolderPath)):
        #     # Add path to list.

        framesPath[index].append(os.path.join(clipsFolderPath, 'frame0.jpg'))
        framesPath[index].append(os.path.join(clipsFolderPath, 'frame1.jpg'))

        framesPath[index].append(os.path.join(clipsFolderPath, 'framet1.jpg'))
        framesPath[index].append(os.path.join(clipsFolderPath, 'framet2.jpg'))
        framesPath[index].append(os.path.join(clipsFolderPath, 'framet3.jpg'))
        framesPath[index].append(os.path.join(clipsFolderPath, 'framet4.jpg'))
        framesPath[index].append(os.path.join(clipsFolderPath, 'framet5.jpg'))
        framesPath[index].append(os.path.join(clipsFolderPath, 'framet6.jpg'))
        framesPath[index].append(os.path.join(clipsFolderPath, 'framet7.jpg'))

        framesPath[index].append(os.path.join(clipsFolderPath, 'frame2.jpg'))
        framesPath[index].append(os.path.join(clipsFolderPath, 'frame3.jpg'))


    # print(framesPath)
    return framesPath



def _pil_loader(path, cropArea=None, resizeDim=None, frameFlip=0):
    with open(path, 'rb') as f:
        img = Image.open(f)
        # Resize image if specified.
        resized_img = img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else img
        # cv2.imwrite(resize)
        # Crop image if crop area specified.
        if cropArea != None:
            cropped_img = resized_img.crop(cropArea)
        else:
            cropped_img = resized_img
        # Flip image horizontally if specified.
        flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else cropped_img


        return flipped_img.convert('RGB')

    
    
class QVI960(data.Dataset):
    def __init__(self, root, transform=None, resizeSize=(640, 360), randomCropSize=(352, 352), train=True):
        # Populate the list with image paths for all the
        # frame in `root`.
        framesPath = _make_dataset(root)
        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))
                
        self.randomCropSize = randomCropSize
        self.cropX0         = resizeSize[0] - randomCropSize[0]
        self.cropY0         = resizeSize[1] - randomCropSize[1]
        self.root           = root
        self.transform      = transform
        self.train          = train
        self.resizeSize     = resizeSize
        self.framesPath     = framesPath

    def __getitem__(self, index):
        sample = []
        inter = None
        if (self.train):
            ### Data Augmentation ###
            # To select random 9 frames from 12 frames in a clip
            firstFrame = 0
            # Apply random crop on the 9 input frames

        

            cropX0 = random.randint(0, self.cropX0)
            cropY0 = random.randint(0, self.cropY0)

            cropArea = (cropX0, cropY0, cropX0 + self.randomCropSize[0], cropY0 + self.randomCropSize[1])

            inter = random.randint(2, 8)

            if (random.randint(0, 1)):
                frameRange = [10, 9, inter, 1, 0]
                inter = 10 - inter
                # returnIndex = IFrameIndex - firstFrame - 1
            else:
                frameRange = [0, 1, inter, 9, 10]
                # returnIndex = firstFrame - IFrameIndex + 7
            # Random flip frame
            randomFrameFlip = random.randint(0, 1)
        else:
            # Fixed settings to return same samples every epoch.
            # For validation/test sets.
            # firstFrame = 0
            cropArea = (0, 0, self.randomCropSize[0], self.randomCropSize[1])
            # IFrameIndex = ((index) % 7  + 1)
            # returnIndex = IFrameIndex - 1
            frameRange = [0, 1, 5, 9, 10]
            randomFrameFlip = 0
            inter = 5
        
        # Loop over for all frames corresponding to the `index`.
        for frameIndex in frameRange:
            # Open image using pil and augment the image.

            image = _pil_loader(self.framesPath[index][frameIndex], cropArea=cropArea,  resizeDim=self.resizeSize, frameFlip=randomFrameFlip)
            # image.save(str(frameIndex) + '.jpg')

            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            sample.append(image)

        # while True:
        #     pass
        t =  (inter - 1.0) / 8.0
        # while True:
        #     pass
        return sample, t


    def __len__(self):
        return len(self.framesPath)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
