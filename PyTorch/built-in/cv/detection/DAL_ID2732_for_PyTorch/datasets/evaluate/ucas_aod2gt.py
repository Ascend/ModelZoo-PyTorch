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
import cv2
import numpy as np
import shutil
import math
from tqdm import tqdm
from decimal import Decimal



def convert_ucas_gt(gt_path, dst_path):
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    os.mkdir(dst_path)
    with open(gt_path,'r') as f:
        files = [x.strip('\n').replace('.png','.txt').replace('AllImages','Annotations') for x in f.readlines()]
    gts = [os.path.split(x)[1] for x in files]
    dst_gt = [os.path.join(dst_path, x) for x in gts]
    print('gt generating...')
    for i, filename in enumerate(tqdm(files)):
        with open(filename,'r',encoding='utf-8-sig') as f:
#             print(filename)
            content = f.read()
            objects = content.split('\n')
            objects = [x for x in objects if len(x)>0]
            for obj in objects:
                name = obj.split()[0]
                box = [ eval(x) for x in  obj.split()[1:9]]
                with open(dst_gt[i],'a') as fd:
                        fd.write('{} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(
                            name,box[0],box[1],box[2],box[3],box[4],box[5],box[6],box[7]
                        ))



if __name__ == "__main__":
    gt_path = '/data-tmp/UCAS_AOD/ma3/UCAS_AOD/test.txt' 
    dst_path = '/data-tmp/UCAS_AOD/ma3/datasets/evaluate/ground-truth'

    convert_ucas_gt(gt_path, dst_path)

    

