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


from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
import cv2
import torch
from multiprocessing import Pool
import glob

array_of_img = [] # this if for store all of the image data
# this function is for read image,the input is directory name
def get_image_paths(folder):
    print(folder)
    res = []
    for j in os.listdir(folder):
        for f in os.listdir(os.path.join(folder, j)):
            if 'jpg' in f:
                s = os.path.join(os.path.join(folder, j),f)
                res.append(s)
    return (a for a in res)

def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    img = cv2.imread(directory_name, cv2.COLOR_BGR2RGB)
    names = directory_name.split('/')
    filename = names[-2]
    f = names[-1]
    im = torch.from_numpy(img).cuda()
    mtcnn = MTCNN(image_size=160, margin=0, device="cuda")
    mtcnn(im, save_path=" "+filename + "/" + f)

rootdir=" "
imgs = get_image_paths(rootdir)
pool = Pool(40)
pool.map(read_directory, imgs)
pool.join()
pool.close()