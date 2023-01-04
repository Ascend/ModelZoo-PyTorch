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
import sys
sys.path.append('./PraNet')
import numpy as np
import torch.nn.functional as F
import torch
import os
from glob import glob
from scipy import misc
from utils.dataloader import test_dataset

testsize = 352

def test(pred_dir, save_path, data_path):
    # 只要res2
    bin_images = glob(os.path.join(pred_dir, '*_3.bin'))
    # 必须要排序，因为原模型代码中输入排序了
    bin_images = sorted(bin_images)

    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, testsize)
    os.makedirs(save_path, exist_ok=True)
    
    for path in bin_images:
        image, gt, name = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)

        res2 = np.fromfile(path, np.float32)
        res = np.reshape(res2, (1, 1, 352, 352))
        res = torch.from_numpy(res)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        misc.imsave(save_path+name, res)
    print("Postprocess Done !")

if __name__ == "__main__":
    data_path = sys.argv[1]
    pred_dir = sys.argv[2]
    save_path = sys.argv[3]
    
    test(pred_dir, save_path,data_path)
