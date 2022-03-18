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
import cv2
import os
import argparse
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import ADNet
from utils import *
from collections import OrderedDict
import torch.distributed as dist

parser = argparse.ArgumentParser(description="ADNet_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='BSD68', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
parser.add_argument("--DeviceID", type=int, default=0, help='choose a device id to use')
parser.add_argument("--demo_img_path", type=str, default='demo_img')
parser.add_argument("--demo_pth_path", type=str, default='data')
opt = parser.parse_args()


def normalize(data):
    return data / 255.
def proc_nodes_module(checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if "module." in k:
            name = k.replace("module.", "")
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

def main():
    # Build model
    local_device = torch.device(f'npu:{opt.DeviceID}')
    torch.npu.set_device(local_device)
    print("using npu :{}".format(opt.DeviceID))
    print('Loading model ...\n')
    net = ADNet(channels=1, num_of_layers=17)
    model = net #model = nn.DataParallel(net, device_ids=device_ids).cuda()
    checkpoint = torch.load(os.path.join(opt.demo_pth_path, 'best_model.pth'), map_location=local_device)
    checkpoint = proc_nodes_module(checkpoint)
    model.load_state_dict(checkpoint)
    model = model.npu()
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.demo_img_path, '*.png'))
    files_source.sort()
    # process data
    psnr_test = 0
    for f in files_source:
        # image
        Img = cv2.imread(f)
        Img = normalize(np.float32(Img[:, :, 0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        # noise
        torch.manual_seed(0)  # set the seed
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / 255.)
        # noisy image
        INoisy = ISource + noise
        ISource = Variable(ISource)
        INoisy = Variable(INoisy)
        ISource = ISource.npu()
        INoisy = INoisy.npu()
        with torch.no_grad():  # this can save much memory
            Out = torch.clamp(model(INoisy), 0., 1.)
        psnr = batch_PSNR(Out, ISource, 1.)
        psnr_test += psnr
        print("%s PSNR %f" % (f, psnr))
        INoisy = INoisy*255
        INoisy = INoisy.data.cpu().numpy()
        INoisy = np.squeeze(INoisy)
        Imag_noise = Image.fromarray(INoisy.astype('uint8'))
        if not os.path.exists('./data/demo_img/result'):
            os.mkdir('./data/demo_img/result')
        Imag_noise.save(os.path.join('data', opt.demo_img_path, 'result', 'image_add_noise.png'))
        print('original image stored in:', os.path.join('data', opt.demo_img_path))
        print('image added noise stored in:', os.path.join('data', opt.demo_img_path, 'result', 'image_add_noise.png'))
        result = Out*255
        result = result.data.cpu().numpy()
        result = np.squeeze(result)
        result = Image.fromarray(result.astype('uint8'))
        result.save(os.path.join('data', opt.demo_img_path, 'result', 'image_after_processing.png'))
        print('image denoised stored in:', os.path.join('data', opt.demo_img_path, 'result', 'image_after_processing.png'))
    psnr_test /= len(files_source)
    print("\nPSNR on demo image %f" % psnr_test)


if __name__ == "__main__":
    main()
