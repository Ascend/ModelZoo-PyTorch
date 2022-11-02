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


def _make_dataset(dir):
    framesPath = []
    # Find and loop over all the clips in root `dir`.

    folders = []

    for index, folder in enumerate(os.listdir(dir)):
        
        clipsFolderPath = os.path.join(dir, folder)
        # Skip items which are not folders.
        if not (os.path.isdir(clipsFolderPath)):
            continue
        # folders[int(folder)] = folder
        folders.append(folder)
    index = 0
    for findex in range(len(sorted(folders))):

        framesPath.append([])
        clipsFolderPath = os.path.join(dir, folders[findex])
        # Find and loop over all the frames inside the clip.
        for image in sorted(os.listdir(clipsFolderPath)):
            # Add path to list.
            framesPath[index].append(os.path.join(clipsFolderPath, image))
        index += 1
    # print(folder)
    return framesPath

def _make_video_dataset(dir):
    framesPath = []
    # Find and loop over all the frames in root `dir`.
    for image in sorted(os.listdir(dir)):
        # Add path to list.
        framesPath.append(os.path.join(dir, image))
    return framesPath

def _pil_loader(path, cropArea=None, resizeDim=None, frameFlip=0):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        # Resize image if specified.
        resized_img = img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else img
        # Crop image if crop area specified.
        cropped_img = img.crop(cropArea) if (cropArea != None) else resized_img
        # Flip image horizontally if specified.
        flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else cropped_img
        return flipped_img.convert('RGB')

    
class Adobe240all(data.Dataset):
    def __init__(self, root, transform=None, resizeSize=(640, 360), randomCropSize=(352, 352), train=True):
        # Populate the list with image paths for all the
        # frame in `root`.
        framesPath = _make_dataset(root)
        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))
        
        dim = resizeSize
        self.randomCropSize = randomCropSize
        self.cropX0         = dim[0] - randomCropSize[0]
        self.cropY0         = dim[1] - randomCropSize[1]
        self.root           = root
        self.transform      = transform
        self.train          = train

        self.framesPath     = framesPath

    def __getitem__(self, index):
        sample = []
        
        if (self.train):
            ### Data Augmentation ###
            # To select random 9 frames from 12 frames in a clip
            firstFrame = 0
            # Apply random crop on the 9 input frames
            cropX = random.randint(0, self.cropX0)
            cropY = random.randint(0, self.cropY0)
            cropArea = (cropX, cropY, cropX + self.randomCropSize[0], cropY + self.randomCropSize[1])
            # Random reverse frame
            #frameRange = range(firstFrame, firstFrame + 9) if (random.randint(0, 1)) else range(firstFrame + 8, firstFrame - 1, -1)
            # IFrameIndex = random.randint(firstFrame + 1, firstFrame + 7)
            # IFrameIndex = firstFrame + 4

            inter = random.randint(9, 15)
            if (random.randint(0, 1)):
                frameRange = [0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24]
                # returnIndex = IFrameIndex - firstFrame - 1
            else:
                frameRange = [24, 16, 15, 14, 13, 12, 11, 10, 9, 8 ,0]
                # returnIndex = firstFrame - IFrameIndex + 7
            # Random flip frame
            randomFrameFlip = random.randint(0, 1)
        else:
            # Fixed settings to return same samples every epoch.
            # For validation/test sets.
            firstFrame = 0
            cropArea = (0, 0, self.randomCropSize[0], self.randomCropSize[1])
            IFrameIndex = ((index) % 7  + 1)
            # returnIndex = IFrameIndex - 1
            frameRange = [0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24]
            randomFrameFlip = 0
        
        # Loop over for all frames corresponding to the `index`.
        for frameIndex in frameRange:
            # Open image using pil and augment the image.
          
            image = _pil_loader(self.framesPath[index][frameIndex], cropArea=cropArea, frameFlip=randomFrameFlip)
            # image.save(str(frameIndex) + '.jpg')
            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            sample.append(image)

        # while True:
        #     pass

        return sample

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
    