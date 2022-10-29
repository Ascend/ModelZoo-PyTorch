# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import cv2
import torchvision.transforms as transforms
import argparse
import torch.nn as nn
import numpy as np
import tqdm
parser = argparse.ArgumentParser(description='Superpoint')
parser.add_argument("--img_path", type=str, default="./preprocess_Result1/", help="result path")
parser.add_argument("--result_path", type=str, default="./preprocess_Result1/", help="result path")
args = parser.parse_args()

if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

def read_image(path):
            input_image = cv2.imread(path)
            #print("image path" + path )
            return input_image

def preprocess(image):
            sizer = [240, 320]
            sizer = np.array(sizer)
            #print(image.shape[:2])
            s = max(sizer /image.shape[:2])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = image[:int(sizer[0]/s), :int(sizer[1]/s)]
            image = cv2.resize(image, (sizer[1], sizer[0]),
                                     interpolation=cv2.INTER_AREA)
            image = image.astype('float32') / 255.0
            if image.ndim == 2:
                image = image[:, :, np.newaxis]
            transform = transforms.Compose([
            transforms.ToTensor(),
        ])
            image = transform(image)
            return image

if __name__ == '__main__':
    with tqdm.tqdm(total=696) as bar:
        for filename in os.listdir(args.img_path):
            for files in os.listdir(os.path.join(args.img_path, filename)):
                if files.endswith(".ppm"):
                    filedir = os.path.join(args.img_path, filename)
                    filedirname = os.path.join(filedir, files)
                    file1 = filedirname.split('/')[3]
                    file2 = filedirname.split('/')[4]
                    file3 = file2.split(".")[0]
                    file4 = file1 + "_" + file3 + ".bin"
                    image = preprocess(read_image(os.path.join(filedir, files)))
                    image1 = image.numpy()
                    #os.mkdir(args.result_path)
                    file5 = os.path.join(args.result_path, file4)
                    image1.tofile(file5)
                    bar.update(1)



    
