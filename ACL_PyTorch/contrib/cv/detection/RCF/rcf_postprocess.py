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
import os
import argparse
import numpy as np
import cv2
from PIL import Image
from scipy.io import savemat


def postprocess_pth(args):
    """[pytorch model postprocess]

    Args:
        args ([argparse]): [pth model postprocess args]
    """
    os.system('rm -rf {}'.format(args.pth_output))
    os.system('mkdir -p {}'.format(args.pth_output))
    assert torch.cuda.is_available(), print('The model must be loaded on GPU')
    device = torch.device("cuda:0")
    model = RCF() # RCF model
    model.to(device)
    checkpoint = torch.load(args.pth_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    class BSDS_dataset(torch.utils.data.Dataset):
        def __init__(self, imgs_dir, h=321, w=481):
            self.imgs_idr = imgs_dir
            img_all_name_list = []
            self.img_name_list = []
            self.h, self.w = h, w
            img_all_name_list = os.listdir(args.imgs_dir)
            for i in range(len(img_all_name_list)):
                img_name = img_all_name_list[i]
                if img_name.endswith(('jpg', 'png', 'jpeg', 'bmp')):
                    img = cv2.imread(os.path.join(self.imgs_idr, img_name)).astype(np.float32)
                    if self.h == img.shape[0] and self.w == img.shape[1]:
                        self.img_name_list.append(img_name)
                        
        def __len__(self):
            return len(self.img_name_list)
        
        def __getitem__(self, index):
            img_name = self.img_name_list[index]
            img = cv2.imread(os.path.join(self.imgs_idr, img_name)).astype(np.float32)
            img -= np.array((104.00698793,116.66876762,122.67891434))
            img = np.transpose(img, (2, 0, 1))
            return img_name, img
    
    h_list, w_list, bs_list = args.height, args.width, args.batch_size
    number = 0
    for k in range(len(h_list)):
        test_dataset = BSDS_dataset(args.imgs_dir, h=h_list[k], w=w_list[k])
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs_list[k])
        for idx, (img_name, image) in enumerate(test_loader):
            image = image.to(device)
            results = model(image) # output is a list, list[0] shape: [b, 1 , h, w]
            result = results[-1].detach().cpu().numpy()
            key = "pth_result"
            for j in range(len(result)):
                save_img_name = img_name[j][:-4]
                save_result = result[j, 0, :, :]
                print(os.path.join(args.imgs_dir, '{}'.format(save_img_name)), "====", number)
                savemat('{}/{}.mat'.format(args.pth_output, save_img_name), {key: save_result})
                number += 1
            
            
def postprocess_om(args):
    """[om model postprocess]

    Args:
        args ([argparse]): [om model postprocess args]
    """
    os.system('rm -rf {}'.format(args.om_output))
    os.system('mkdir -p {}'.format(args.om_output))
    img_name_list = os.listdir(args.imgs_dir)
    for k in range(len(img_name_list)):
        img_name = img_name_list[k]
        if img_name.endswith(('jpg', 'png', 'jpeg', 'bmp')):
            print(os.path.join(args.imgs_dir, '{}'.format(img_name)), "====", k)
            img = cv2.imread(os.path.join(args.imgs_dir, '{}'.format(img_name)))
            h, w, c = img.shape
            # Read the output file of the om model
            img_out = np.fromfile('{}/{}_6.bin'.format(args.bin_dir, img_name[:-4]), dtype="float32")
            img_out = img_out.reshape((h, w))
            key = "om_result"
            savemat('{}/{}.mat'.format(args.om_output, img_name[:-4]), {key: img_out})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rcf postprocess') # rcf postprocess parameters
    parser.add_argument('--model', default='om',
                        type=str, help='om model or pth model')
    parser.add_argument('--imgs_dir', default='data/BSR/BSDS500/data/images/test',
                        type=str, help='images path')
    parser.add_argument('--bin_dir', default='result/dumpOutput_device0',
                        type=str, help='bin file path inferred by benchmark')
    parser.add_argument('--pth_path', default='RCF-pytorch/RCFcheckpoint_epoch12.pth',
                        type=str, help='pth path')
    parser.add_argument('--om_output', default='data/om_out',
                        type=str, help='om postprocess output dir')
    parser.add_argument('--pth_output', default='data/pth_out',
                        type=str, help='pth postprocess output dir')
    parser.add_argument('--batch_size', nargs='+',
                        type=int, help='batch size')
    parser.add_argument('--height', nargs='+',
                        type=int, help='input height')
    parser.add_argument('--width', nargs='+',
                        type=int, help='input width')
    args = parser.parse_args()
    
    if args.model == 'om':
        postprocess_om(args)
    else:
        sys.path.append('./RCF-pytorch')
        from models import RCF
        import torch
        postprocess_pth(args)
  