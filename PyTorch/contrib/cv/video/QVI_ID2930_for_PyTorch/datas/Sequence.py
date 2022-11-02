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

def _make_dataset(dir, inter_frames):
    framesPath = []
    framesIndex = []
    framesFolder = []
    # Find and loop over all the clips in root `dir`.

    totindex = 0

    for folder in os.listdir(dir):
        
        clipsFolderPath = os.path.join(dir, folder)
        # Skip items which are not folders.
        if not (os.path.isdir(clipsFolderPath)):
            continue


        # Find and loop over all the frames inside the clip.
        frames = sorted(os.listdir(clipsFolderPath))

        group_len = (len(frames) - 1) // (inter_frames + 1) - 2
        for index in range(group_len):
            framesPath.append([])
            framesIndex.append([])
            framesFolder.append([])
            # Add path to list.

            # frame 0
            framesFolder[totindex].append(folder)
            framesPath[totindex].append(os.path.join(clipsFolderPath, frames[index * (inter_frames + 1)]))
            framesIndex[totindex].append(frames[index * (inter_frames + 1)][:-4])

            # frame 1 .... frame 2
            for ii in range (0, inter_frames + 2):
                framesFolder[totindex].append(folder)
                framesPath[totindex].append(os.path.join(clipsFolderPath, frames[(index + 1) * (inter_frames + 1) + ii]))
                framesIndex[totindex].append(frames[(index + 1) * (inter_frames + 1) + ii][:-4])

            # frame 3
            framesFolder[totindex].append(folder)
            framesPath[totindex].append(os.path.join(clipsFolderPath, frames[(index + 3)* (inter_frames + 1)]))
            framesIndex[totindex].append(frames[(index + 3) * (inter_frames + 1)][:-4])

            totindex += 1
    # print(folder)
    return framesPath, framesFolder, framesIndex


def _pil_loader(path, cropArea=None, resizeDim=None, frameFlip=0):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        # Resize image if specified.
        resized_img = img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else img
        # Crop image if crop area specified.
        cropped_img = resized_img.crop(cropArea) if (cropArea != None) else resized_img
        # Flip image horizontally if specified.
        flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else cropped_img
        return flipped_img.convert('RGB')

    
class Sequence(data.Dataset):
    def __init__(self, root, transform=None, resizeSize=(640, 360), randomCropSize=(352, 352), inter_frames=7):
        # Populate the list with image paths for all the
        # frame in `root`.
        framesPath, framesFolder, framesIndex = _make_dataset(root, inter_frames)

        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))
        
        self.dim = resizeSize
        self.randomCropSize = randomCropSize
        self.cropX0         = self.dim[0] - randomCropSize[0]
        self.cropY0         = self.dim[1] - randomCropSize[1]
        self.root           = root
        self.transform      = transform
      
        self.inter_frames   = inter_frames
        self.framesPath     = framesPath
        self.framesFolder   = framesFolder
        self.framesIndex     = framesIndex

    def __getitem__(self, index):
        sample = []
        folders = []
        indeces = []

        cropArea = (0, 0, self.randomCropSize[0], self.randomCropSize[1])
        IFrameIndex = ((index) % 7  + 1)
            # returnIndex = IFrameIndex - 1
        frameRange = [0]
        for ii in range(self.inter_frames + 2):
            frameRange.append(ii + 1)
        frameRange.append(self.inter_frames + 3)

        randomFrameFlip = 0
        # print(frameRange)
        # print(index)
        # Loop over for all frames corresponding to the `index`.
        for frameIndex in frameRange:
            # Open image using pil and augment the image.
            # print(frameIndex)
            image = _pil_loader(self.framesPath[index][frameIndex], cropArea=cropArea, resizeDim=self.dim, frameFlip=randomFrameFlip)
            folder = self.framesFolder[index][frameIndex]
            iindex = self.framesIndex[index][frameIndex]
            
            # image.save(str(frameIndex) + '.jpg')
            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            sample.append(image)
            indeces.append(iindex)
            folders.append(folder)


        # while True:
        #     pass

        return sample, folders, indeces

    def __len__(self):
        """
        Returns the size of dataset. Invoked as len(datasetObj).

        Returns
        -------
            int
                number of samples.
        """


        return len(self.framesPath)

    def __repr__(self):
        """
        Returns printable representation of the dataset object.

        Returns
        -------
            string
                info.
        """


        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
